import sys
import math
import argparse
import tensorflow as tf
import tensorflow.models.rnn.rnn as rnn
import numpy as np
from collections import namedtuple
from itertools import izip

class Vocabulary:
  def __init__(self):
    self.w2i = {}
    self.i2w = []
  def Convert(self, w):
    if isinstance(w, int):
      assert w >= 0 and w < len(self.iw2)
      return self.i2w[w]
    else:
      if not w in self.w2i:
        self.w2i[w] = len(self.i2w)
        self.i2w.append(w)
      return self.w2i[w]
  def __len__(self):
    return len(self.i2w)

def ReadCorpus(filename, vocab):
  corpus = []
  with open(filename) as f:
    for line in f:
      words = line.split()
      words = [vocab.Convert(word) for word in words]
      words.append(vocab.Convert('</s>'))
      corpus.append(words)
  return corpus

Token = namedtuple('Token', 'word, morphemes, chars')
def ReadMorphCorpus(filename, word_vocab, morpheme_vocab, char_vocab):
  corpus = []
  current_sentence = []
  with open(filename) as f:
    for line in f:
      if not line.strip():
        assert len(current_sentence) > 0
        corpus.append(current_sentence)
        del current_sentence[:]
      else:
        word, analyses = line.decode('utf-8').strip().split('\t', 1)
        token_word = word_vocab.Convert(word)
        token_chars = [char_vocab.Convert(c) for c in word]
        token_morphemes = []
        for analysis in analyses.split('\t'):
          token_morphemes.append([morpheme_vocab.Convert(m) for m in analysis.split('+')])
        current_sentence.append(Token(token_word, token_morphemes, token_chars))
  assert len(current_sentence) == 0
  return corpus

def pad(sentence, vocab, length):
  return sentence + max(0, length - len(sentence)) * [vocab.Convert('<pad>')]

def init_random_normal(dims):
  stddev = 1.0 / math.sqrt(sum(dims))
  return tf.random_normal(dims, stddev=stddev)

class WordEmbedder:
  def __init__(self, vocab_size, embedding_dim):
    self.embedding_matrix = tf.Variable(init_random_normal([vocab_size, embedding_dim]))

  def embed(self, sentence):
    return tf.nn.embedding_lookup(self.embedding_matrix, sentence)

class EncoderModel:
  def __init__(self, max_length, hidden_dim, lstm_layer_count, batch_size, source_vocab_size, embedding_dim):
    self.word_embedder = WordEmbedder(source_vocab_size, embedding_dim)

    self.fwd_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
    self.rev_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
    self.fwd_encoder = tf.nn.rnn_cell.MultiRNNCell(self.fwd_encoder_cells * lstm_layer_count)
    self.rev_encoder = tf.nn.rnn_cell.MultiRNNCell(self.rev_encoder_cells * lstm_layer_count)
    self.fwd_initial_state = self.fwd_encoder.zero_state(batch_size, tf.float32)
    self.rev_initial_state = self.rev_encoder.zero_state(batch_size, tf.float32)

  def embed_source(self, source_sentence, source_word_embeddings):
    return self.word_embedder.embed(source_sentence)

  def build_annotations(self, encoder_inputs, source_lengths):
    input_embs = self.word_embedder.embed(encoder_inputs)
    input_embs_list = tf.unpack(tf.transpose(input_embs, perm=[1, 0, 2]))
    annotations, fwd_state, rev_state = rnn.bidirectional_rnn(self.fwd_encoder, self.rev_encoder, \
      input_embs_list, self.fwd_initial_state, self.rev_initial_state, sequence_length=source_lengths)
    annotation_matrix = tf.pack(annotations) # (max_length, ?, 2*hidden_dim)
    return annotation_matrix, fwd_state, rev_state

class DecoderModel:
  def __init__(self, embedding_dim, target_vocab_size, hidden_dim, lstm_layer_count, max_length):
    # Word Embeddings
    self.bos_word_emb = tf.Variable(init_random_normal([embedding_dim])) # Used when generating target output
    self.target_word_emb = tf.Variable(init_random_normal([len(target_vocab), embedding_dim]))

    self.decoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
    self.decoder = tf.nn.rnn_cell.MultiRNNCell(self.decoder_cells * lstm_layer_count)

    # Used to transform the reverse encoder's LSTM state into the decoder's initial state
    self.transform_W = tf.Variable(init_random_normal([self.decoder.state_size, 2 * hidden_dim]))
    self.transform_b = tf.Variable(tf.zeros([self.decoder.state_size]))

    # Used to transform the output of the decoder LSTM into a distribution over target vocabulary
    self.final_W = tf.Variable(init_random_normal([hidden_dim, target_vocab_size]))
    self.final_b = tf.Variable(tf.zeros([target_vocab_size]))

    self.hidden_dim = hidden_dim
    self.target_vocab_size = target_vocab_size
    self.max_length = max_length

  def create_initial_state(self, rev_state):
    return tf.matmul(rev_state, self.transform_W) + self.transform_b

  def add_input(self, input_tensor, state):
    return self.decoder(input_tensor, state)

  def embed(self, decoder_inputs):
    return tf.nn.embedding_lookup(self.target_word_emb, decoder_inputs)

  def compute_output_distributions(self, decoder_outputs):
    # We want to multiply this tensor by final_W, which is (hidden_dim, target_vocab_size).
    # This is equivalent to doing a matrix multiply (?*max_length, hidden_dim) times (hidden_dim, target_vocab_size)
    # Since TF doesn't let us do tensor-matrix products, we use this transformation as a hackaround.
    decoder_outputs_t = tf.transpose(decoder_outputs, perm=[1, 0, 2]) # (max_length, ?, hidden_dim)
    decoder_outputs_tr = tf.reshape(decoder_outputs_t, [-1, self.hidden_dim]) # (max_length * ?, hidden_dim)
    output_dists = tf.matmul(decoder_outputs_tr, self.final_W) + self.final_b # (max_length * ?, target_vocab_size)
    return tf.reshape(output_dists, [self.max_length, -1, self.target_vocab_size])

  @property
  def state_size(self):
    return self.decoder.state_size

class AttentionModel:
  def __init__(self, alignment_hidden_dim, hidden_dim, output_state_size, max_length):
    # Used to compute alignment/attention
    self.alignment_U = tf.Variable(tf.random_normal([alignment_hidden_dim, 1]))
    self.alignment_W = tf.Variable(tf.random_normal([2 * hidden_dim, alignment_hidden_dim]))
    self.alignment_V = tf.Variable(tf.random_normal([decoder.state_size, alignment_hidden_dim]))
    self.alignment_b = tf.Variable(tf.zeros([alignment_hidden_dim]))

    self.alignment_hidden_dim = alignment_hidden_dim
    self.hidden_dim = hidden_dim
    self.output_state_size = output_state_size
    self.max_length = max_length

  def compute_attention(self, output_state, annotation_matrix):
    # The attention vector A should have dimensions (?, max_length)
    # and is generated from the annotation matrix I (max_length, ?, 2*hidden_dim) => (max_length * ?, 2*hidden_dim)
    # and the output state S (?, hidden_dim).
    # We want A = UH, H = tanh(WI + VS + b)
    # Where H is (?, max_length, alignment_hidden_dim)

    # \Alpha is the normalized version of A: \Alpha = softmax(A)
    annotation_matrix_r = tf.reshape(annotation_matrix, [-1, 2 * self.hidden_dim]) # (?*max_length, 2*hidden_dim)
    bias_term = tf.matmul(output_state, self.alignment_V) + self.alignment_b # [?, alignment_hidden_dim], will be broadcast to [S, ?, alignment_hidden_dim]
    IW = tf.matmul(annotation_matrix_r, self.alignment_W)
    IW_r = tf.reshape(IW, [self.max_length, -1, self.alignment_hidden_dim])
    alignment_hidden = tf.nn.tanh(IW_r + bias_term) # [S, ?, alignment_hidden_dim]
    H_r = tf.reshape(alignment_hidden, [-1, self.alignment_hidden_dim])
    alignment = tf.matmul(H_r, self.alignment_U)
    attention = tf.nn.softmax(alignment) # [S * ?, 1]
    attention_r = tf.reshape(attention, [self.max_length, -1, 1]) # [S, ?, 1]
    return attention_r

  def compute_context(self, annotation_matrix, attention):
    # We essentially want I^T * A, but the batch sizes get in the way.
    # We solve this by changing the dot product into two steps:
    # an element-wise product, and a summation.
    IA = tf.mul(annotation_matrix, attention)
    return tf.reduce_sum(IA, 0) # [?, 2H]

parser = argparse.ArgumentParser()
parser.add_argument('source_filename')
parser.add_argument('target_filename')
parser.add_argument('--morph', required=False)
args = parser.parse_args()

tf.set_random_seed(1)
if args.morph:
  word_vocab = Vocabulary()
  morph_vocab = Vocabulary()
  char_vocab = Vocabulary()
  ReadMorphCorpus(sys.argv[1], word_vocab, morph_vocab, char_vocab)
  sys.exit(1)

source_vocab = Vocabulary()
target_vocab = Vocabulary()
source_vocab.Convert('<pad>')
target_vocab.Convert('<pad>')
source_corpus = ReadCorpus(args.source_filename, source_vocab)
target_corpus = ReadCorpus(args.target_filename, target_vocab)
# We could pad the whole corpus here, rather than doing it over and over
assert len(source_corpus) == len(target_corpus)
print >>sys.stderr, 'Vocab sizes: %d/%d' % (len(source_vocab), len(target_vocab))

minibatch_size = 1
embedding_dim = 7
hidden_dim = 13
alignment_hidden_dim = 4
lstm_layer_count = 1
max_source_length = max(len(sent) for sent in source_corpus)
max_target_length = max(len(sent) for sent in target_corpus)
print >>sys.stderr, 'Max lengths: %d/%d' % (max_source_length, max_target_length)

sess = tf.Session()

encoder_inputs = tf.placeholder(tf.int32, shape=[None, max_source_length])
decoder_inputs = tf.placeholder(tf.int32, shape=[None, max_target_length])
source_lengths = tf.placeholder(tf.int32, shape=[None])
target_lengths = tf.placeholder(tf.int32, shape=[None])
batch_size = tf.placeholder(tf.int32, shape=[])

encoder = EncoderModel(max_source_length, hidden_dim, lstm_layer_count, batch_size, len(source_vocab), embedding_dim)
decoder = DecoderModel(embedding_dim, len(target_vocab), hidden_dim, lstm_layer_count, max_target_length)
attention_model = AttentionModel(alignment_hidden_dim, hidden_dim, decoder.state_size, max_source_length)

annotation_matrix, fwd_state, rev_state = encoder.build_annotations(encoder_inputs, source_lengths)

state = decoder.create_initial_state(rev_state)
prev_word = tf.reshape(tf.tile(tf.expand_dims(decoder.bos_word_emb, 0), tf.pack([batch_size, 1])), [-1, embedding_dim])

decoder_input_embs = decoder.embed(decoder_inputs)
# This is basically just an unrolled version of dynamic_rnn, which gives us access to te
# states and outputs one at a time, so we can compute attention and whatnot.
input_embs_t = tf.transpose(decoder_input_embs, perm=[1, 0, 2])
outputs = []
for t, input_emb in enumerate(tf.unpack(input_embs_t)):
  if t > 0:
    tf.get_variable_scope().reuse_variables()

  attention = attention_model.compute_attention(state, annotation_matrix)
  context = attention_model.compute_context(annotation_matrix, attention)

  prev_word_and_context = tf.concat(1, [tf.reshape(prev_word, [-1, embedding_dim]), context]) 
 
  output, state = decoder.add_input(prev_word_and_context, state) # Output should be (?, H)
  outputs.append(output)
  prev_word = input_emb

decoder_outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2]) # (?, T, H)
output_dists = decoder.compute_output_distributions(decoder_outputs)

# We reshape both the output distributions and the references to be (max_length*?) dimensional
# and compute the loss in that space
output_dists_r = tf.reshape(output_dists, [-1, len(target_vocab)])
decoder_inputs_r = tf.reshape(decoder_inputs, [-1])
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output_dists_r, decoder_inputs_r)
total_xent = tf.reduce_sum(cross_entropy)

#optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(total_xent)
sess.run(tf.initialize_all_variables())

def construct_feed_dict(source_batch, target_batch):
    global source_vocab
    global target_vocab
    global max_source_length
    global max_target_length

    source_inputs = []
    target_inputs = []
    src_lengths = []
    tgt_lengths = []
    N = max(len(sentence) for sentence in source_batch)
    M = max(len(sentence) for sentence in target_batch)
    for source_sentence, target_sentence in izip(source_batch, target_batch):
      padded_source = pad(source_sentence, source_vocab, max_source_length)
      padded_target = pad(target_sentence, target_vocab, max_target_length)
      source_inputs.append(padded_source)
      target_inputs.append(padded_target)
      src_lengths.append(len(source_sentence))
      tgt_lengths.append(len(target_sentence))

    feed_dict = {}
    feed_dict[batch_size.name] = len(source_inputs)
    feed_dict[source_lengths.name] = np.array(src_lengths)
    feed_dict[target_lengths.name] = np.array(tgt_lengths)
    feed_dict[encoder_inputs.name] = np.array(source_inputs)
    feed_dict[decoder_inputs.name] = np.array(target_inputs) 

    return feed_dict

total_target_words = sum(len(sent) for sent in target_corpus)
for i in range(100000):
  report_loss = 0.0
  total_loss = 0.0
  report_target_words = 0
  for j in range(0, len(source_corpus), minibatch_size):
    J = min(j + minibatch_size, len(source_corpus))
    feed_dict = construct_feed_dict(source_corpus[j:J], target_corpus[j:J])
    target_words = sum(len(sentence) for sentence in target_corpus[j:J])
    output, loss = sess.run([train_step, total_xent], feed_dict=feed_dict)
    total_loss += loss
    report_loss += loss
    report_target_words += target_words
    if (report_target_words - target_words) // 1000 != report_target_words // 1000:
      print >>sys.stderr, 'Parital perp:', math.exp(report_loss / report_target_words)
      report_target_words = 0
      report_loss = 0.0
  print 'Perplexity:', math.exp(total_loss / total_target_words), '(%f loss over %d words)' % (total_loss, total_target_words)
