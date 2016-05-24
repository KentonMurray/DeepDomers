import sys
import math
import argparse
import tensorflow as tf
import tensorflow.models.rnn.rnn as rnn
import numpy as np
from collections import namedtuple
from itertools import izip

debug_tensor = None

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

Token = namedtuple('Token', 'word, analyses, chars')
def ReadMorphCorpus(filename, word_vocab, morpheme_vocab, char_vocab):
  corpus = []
  current_sentence = []
  eos_word = word_vocab.Convert('</s>')
  eos = Token(eos_word, [], [])
  with open(filename) as f:
    for line in f:
      if not line.strip():
        assert len(current_sentence) > 0
        current_sentence.append(eos)
        corpus.append(current_sentence)
        current_sentence = []
      else:
        word, analyses = line.decode('utf-8').strip().split('\t', 1)
        token_word = word_vocab.Convert(word)

        token_chars = [char_vocab.Convert(c) for c in word]
        token_chars.append(char_vocab.Convert('</w>'))

        token_analyses = []
        for analysis in analyses.split('\t'):
          analysis = [morpheme_vocab.Convert(m) for m in analysis.split('+')]
          analysis.append(morpheme_vocab.Convert('</w>'))
          token_analyses.append(analysis)

        current_sentence.append(Token(token_word, token_analyses, token_chars))

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

class CharacterEmbedder:
  def __init__(self, char_vocab_size, char_embedding_dim, word_embedding_dim, lstm_layer_count, batch_size):
    rev_output_dim = word_embedding_dim // 2
    fwd_output_dim = (word_embedding_dim - rev_output_dim)
    self.char_embeddings = tf.Variable(init_random_normal([char_vocab_size, char_embedding_dim]))
    self.fwd_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(fwd_output_dim) for _ in range(lstm_layer_count)]
    self.rev_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(rev_output_dim) for _ in range(lstm_layer_count)]
    self.fwd_encoder = tf.nn.rnn_cell.MultiRNNCell(self.fwd_encoder_cells)
    self.rev_encoder = tf.nn.rnn_cell.MultiRNNCell(self.rev_encoder_cells)
    self.fwd_initial_state = self.fwd_encoder.zero_state(batch_size, tf.float32)
    self.rev_initial_state = self.rev_encoder.zero_state(batch_size, tf.float32)
    self.char_embedding_dim = char_embedding_dim
    self.word_embedding_dim = word_embedding_dim

  def embed(self, source_chars, sent_lengths, word_lengths, max_sent_length, max_word_length):
    global debug_tensor
    # source_chars is [?, max_sent_len, max_word_len]
    # sent_lengths is [?]
    # word_lengths is [?, max_sent_len]

    word_lengths_r = tf.to_int64(tf.reshape(word_lengths, [-1]))
    input_embs = tf.nn.embedding_lookup(self.char_embeddings, source_chars) # [?, max_sent_len, max_word_len, char_emb_dim]
    input_embs_r = tf.reshape(input_embs, [-1, max_word_length, self.char_embedding_dim]) # [? * max_sent_len, max_word_len, char_emb_dim]
    input_embs_rt = tf.transpose(input_embs_r, perm=[1, 0, 2]) # [max_word_len, ? * max_sent_len, char_emb_dim]
    input_embs_rt_rev = tf.reverse_sequence(input_embs_rt, seq_lengths=word_lengths_r, seq_dim=0, batch_dim=1)

    # TODO: We should pass ini initial_state=self.fwd_initial_state (or rev_initial_state)
    # but for some inexplicable reason this code crashes if we do...
    with tf.variable_scope('fwd'):
      forward_encodings, _ = rnn.dynamic_rnn(self.fwd_encoder, input_embs_rt, sequence_length=word_lengths_r, dtype=tf.float32, time_major=True) # [max_word_len, ?, word_emb_dim]
    with tf.variable_scope('rev'):
      reverse_encodings, _ = rnn.dynamic_rnn(self.rev_encoder, input_embs_rt_rev, sequence_length=word_lengths_r, dtype=tf.float32, time_major=True)

    # We really want the last encoding from each sequence, which will be the last char for the fwd
    # and the first char for the rev (since we reversed its input). To get this, we reverse both
    # encoding tensors and just take the 0th column of their concatenation
    forward_encodings_rev = tf.reverse_sequence(forward_encodings, seq_lengths=word_lengths_r, seq_dim=0, batch_dim=1) # [max_word_len, ? * max_sent_len, word_emb_dim]
    reverse_encodings_rev = tf.reverse_sequence(reverse_encodings, seq_lengths=word_lengths_r, seq_dim=0, batch_dim=1) # [max_word_len, ? * max_sent_len, word_emb_dim]
    final_encodings_rev = tf.concat(2, [forward_encodings_rev, reverse_encodings_rev]) # [max_word_len, ? * max_sent_len, 2 * word_emb_dim]
    final_encodings_0th = tf.slice(final_encodings_rev, [1, 0, 0], [-1, -1, -1]) # [1, ? * max_sent_len, 2 * word_emb_dim]
    final_encodings_0th_s = tf.squeeze(final_encodings_0th, squeeze_dims=[0]) # [? * max_sent_len, 2 * word_emb_dim]
    final_encodings_0th_sr = tf.reshape(final_encodings_0th_s, [-1, max_sent_length, self.word_embedding_dim]) # [?, max_sent_len, 2 * word_emb_dim]
    debug_tensor = final_encodings_0th_sr
    return final_encodings_0th_sr 

class EncoderModel:
  def __init__(self, max_length, hidden_dim, lstm_layer_count, batch_size, source_vocab_size, embedding_dim):
    self.word_embedder = WordEmbedder(source_vocab_size, embedding_dim)
    self.char_embedder = CharacterEmbedder(100, 16, embedding_dim, lstm_layer_count, batch_size)

    self.fwd_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for _ in range(lstm_layer_count)]
    self.rev_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for _ in range(lstm_layer_count)]
    self.fwd_encoder = tf.nn.rnn_cell.MultiRNNCell(self.fwd_encoder_cells)
    self.rev_encoder = tf.nn.rnn_cell.MultiRNNCell(self.rev_encoder_cells)
    self.fwd_initial_state = self.fwd_encoder.zero_state(batch_size, tf.float32)
    self.rev_initial_state = self.rev_encoder.zero_state(batch_size, tf.float32)

  def embed_source(self, source_sentence):
    return self.word_embedder.embed(source_sentence)

  def embed_source_from_chars(self, source_chars, sent_lengths, word_lengths, max_sent_length, max_word_length):
    return self.char_embedder.embed(source_chars, sent_lengths, word_lengths, max_sent_length, max_word_length)

  def build_annotations(self, encoder_inputs, source_lengths):
    input_embs = self.embed_source(encoder_inputs) # [?, max_length, word_emb_dim]
    input_embs_list = tf.unpack(tf.transpose(input_embs, perm=[1, 0, 2]))
    annotations, fwd_state, rev_state = rnn.bidirectional_rnn(self.fwd_encoder, self.rev_encoder, \
      input_embs_list, self.fwd_initial_state, self.rev_initial_state, sequence_length=source_lengths)
    annotation_matrix = tf.pack(annotations) # (max_length, ?, 2*hidden_dim)
    return annotation_matrix, fwd_state, rev_state

  def build_annotations_from_chars(self, encoder_word_inputs, encoder_char_inputs, sent_lengths, word_lengths, max_sent_length, max_word_length):
    input_word_embs = self.embed_source(encoder_word_inputs)
    input_char_embs = self.embed_source_from_chars(encoder_char_inputs, sent_lengths, word_lengths, max_sent_length, max_word_length) # [?, max_length, word_emb_dim]
    input_embs = tf.concat(2, [input_word_embs, input_char_embs])
    input_embs_list = tf.unpack(tf.transpose(input_embs, perm=[1, 0, 2]))
    annotations, fwd_state, rev_state = rnn.bidirectional_rnn(self.fwd_encoder, self.rev_encoder, \
      input_embs_list, self.fwd_initial_state, self.rev_initial_state, sequence_length=sent_lengths)
    annotation_matrix = tf.pack(annotations) # (max_length, ?, 2*hidden_dim)
    return annotation_matrix, fwd_state, rev_state

class DecoderModel:
  def __init__(self, embedding_dim, target_vocab_size, hidden_dim, lstm_layer_count, max_length):
    # Word Embeddings
    self.bos_word_emb = tf.Variable(init_random_normal([embedding_dim])) # Used when generating target output
    self.target_word_emb = tf.Variable(init_random_normal([target_vocab_size, embedding_dim]))

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

  def compute_loss(self, outputs, references):
    # outputs should be [?, T, H]
    # references should be [?, T]
    output_distributions = self.compute_output_distributions(outputs)

    # We reshape both the output distributions and the references to be (max_length*?) dimensional
    # and compute the loss in that space
    output_dists_r = tf.reshape(output_distributions, [-1, len(target_word_vocab)])
    references_r = tf.reshape(references, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output_dists_r, references_r)
    total_xent = tf.reduce_sum(cross_entropy)
    return total_xent

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

source_word_vocab = Vocabulary()
target_word_vocab = Vocabulary()
source_morpheme_vocab = Vocabulary()
target_morpheme_vocab = Vocabulary()
source_char_vocab = Vocabulary()
target_char_vocab = Vocabulary()
source_vocabs = [source_word_vocab, source_morpheme_vocab, source_char_vocab]
target_vocabs = [target_word_vocab, target_morpheme_vocab, target_char_vocab]
vocabs = source_vocabs + target_vocabs
for vocab in vocabs:
  vocab.Convert('<pad>')
source_corpus = ReadMorphCorpus(args.source_filename, *source_vocabs)
target_corpus = ReadMorphCorpus(args.target_filename, *target_vocabs)
# We could pad the whole corpus here, rather than doing it over and over
assert len(source_corpus) == len(target_corpus)
print >>sys.stderr, 'Source vocab sizes: %d words, %d morphemes, %d characters' % (len(source_word_vocab), len(source_morpheme_vocab), len(source_word_vocab))
print >>sys.stderr, 'Target vocab sizes: %d words, %d morphemes, %d characters' % (len(target_word_vocab), len(target_morpheme_vocab), len(target_word_vocab))

minibatch_size = 1
embedding_dim = 7
hidden_dim = 13
alignment_hidden_dim = 4
lstm_layer_count = 1
print source_corpus[0]
max_source_sent_length = max(len(sent) for sent in source_corpus)
max_target_sent_length = max(len(sent) for sent in target_corpus)
max_source_analyses = max(len(token.analyses) for sent in source_corpus for token in sent)
max_target_analyses = max(len(token.analyses) for sent in target_corpus for token in sent)
max_source_analysis_length = max(len(analysis) for sent in source_corpus for token in sent for analysis in token.analyses)
max_target_analysis_length = max(len(analysis) for sent in target_corpus for token in sent for analysis in token.analyses)
max_source_word_length = max(len(token.chars) for sent in source_corpus for token in sent)
max_target_word_length = max(len(token.chars) for sent in target_corpus for token in sent)
print >>sys.stderr, 'Sentences have at most %d/%d tokens' % (max_source_sent_length, max_target_sent_length)
print >>sys.stderr, 'Words have at most %d/%d analyses' % (max_source_analyses, max_target_analyses)
print >>sys.stderr, 'Analyses have at most %d/%d morphemes' % (max_source_analysis_length, max_target_analysis_length)
print >>sys.stderr, 'Words have at most %d/%d characters' % (max_source_word_length, max_target_word_length)

sess = tf.Session()

source_word_inputs = tf.placeholder(tf.int32, shape=[None, max_source_sent_length])
source_morpheme_inputs = tf.placeholder(tf.int32, shape=[None, max_source_sent_length * max_source_analyses * max_source_analysis_length])
source_character_inputs = tf.placeholder(tf.int32, shape=[None, max_source_sent_length, max_source_word_length])

target_word_inputs = tf.placeholder(tf.int32, shape=[None, max_target_sent_length])
target_morpheme_inputs = tf.placeholder(tf.int32, shape=[None, max_target_sent_length * max_target_analyses * max_target_analysis_length])
target_character_inputs = tf.placeholder(tf.int32, shape=[None, max_target_sent_length * max_target_word_length])

source_sent_lengths = tf.placeholder(tf.int32, shape=[None])
source_analysis_counts = tf.placeholder(tf.int32, shape=[None, max_source_sent_length])
source_analysis_lengths = tf.placeholder(tf.int32, shape=[None, max_source_sent_length * max_source_analyses])
source_word_lengths = tf.placeholder(tf.int32, shape=[None, max_source_sent_length])

target_sent_lengths = tf.placeholder(tf.int32, shape=[None])
target_analysis_counts = tf.placeholder(tf.int32, shape=[None, max_target_sent_length])
target_analysis_lengths = tf.placeholder(tf.int32, shape=[None, max_target_sent_length * max_target_analyses])
target_word_lengths = tf.placeholder(tf.int32, shape=[None, max_target_sent_length])

batch_size = tf.placeholder(tf.int32, shape=[])

encoder = EncoderModel(max_source_sent_length, hidden_dim, lstm_layer_count, batch_size, len(source_word_vocab), embedding_dim)
decoder = DecoderModel(embedding_dim, len(target_word_vocab), hidden_dim, lstm_layer_count, max_target_sent_length)
attention_model = AttentionModel(alignment_hidden_dim, hidden_dim, decoder.state_size, max_source_sent_length)

annotation_matrix, fwd_state, rev_state = encoder.build_annotations_from_chars(source_word_inputs, source_character_inputs, source_sent_lengths, source_word_lengths, max_source_sent_length, max_source_word_length)

state = decoder.create_initial_state(rev_state)
prev_word = tf.reshape(tf.tile(tf.expand_dims(decoder.bos_word_emb, 0), tf.pack([batch_size, 1])), [-1, embedding_dim])

decoder_input_embs = decoder.embed(target_word_inputs)
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
total_xent = decoder.compute_loss(decoder_outputs, target_word_inputs)

#optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(total_xent)
sess.run(tf.initialize_all_variables())

def construct_feed_dict(source_batch, target_batch):
    global source_vocab
    global target_vocab
    global max_source_length
    global max_target_length

    source_input_words = []
    target_input_words = []
    source_input_chars = []
    src_word_lengths = []
    src_lengths = []
    tgt_lengths = []
    N = max(len(sentence) for sentence in source_batch)
    M = max(len(sentence) for sentence in target_batch)
    for source_sentence, target_sentence in izip(source_batch, target_batch):
      src_lengths.append(len(source_sentence))
      tgt_lengths.append(len(target_sentence))

      padded_source_words = pad([token.word for token in source_sentence], source_word_vocab, max_source_sent_length)
      padded_target_words = pad([token.word for token in target_sentence], target_word_vocab, max_target_sent_length)
      source_input_words.append(padded_source_words)
      target_input_words.append(padded_target_words)

      padded_source_characters = []
      src_word_lengths.append([])
      for i in range(max_source_sent_length):
        if i < len(source_sentence):
          padded_source_characters.append(pad(source_sentence[i].chars, source_char_vocab, max_source_word_length))
          src_word_lengths[-1].append(len(source_sentence[i].chars))
        else:
          padded_source_characters.append(pad([], source_char_vocab, max_source_word_length))
          src_word_lengths[-1].append(0)
      source_input_chars.append(padded_source_characters)


    feed_dict = {}
    feed_dict[batch_size.name] = len(source_input_words)
    feed_dict[source_sent_lengths.name] = np.array(src_lengths)
    feed_dict[target_sent_lengths.name] = np.array(tgt_lengths)
    feed_dict[source_word_inputs.name] = np.array(source_input_words)
    feed_dict[target_word_inputs.name] = np.array(target_input_words)
    feed_dict[source_character_inputs.name] = np.array(source_input_chars) 
    feed_dict[source_word_lengths.name] = np.array(src_word_lengths) 

    return feed_dict

total_target_words = sum(len(sent) for sent in target_corpus)
for i in range(100000):
  report_loss = 0.0
  total_loss = 0.0
  report_target_words = 0
  report_frequency = 100
  for j in range(0, len(source_corpus), minibatch_size):
    J = min(j + minibatch_size, len(source_corpus))
    feed_dict = construct_feed_dict(source_corpus[j:J], target_corpus[j:J])
    target_words = sum(len(sentence) for sentence in target_corpus[j:J])

    """debug = sess.run([debug_tensor], feed_dict=feed_dict)
    for thing in debug:
      print thing.shape
      print thing
    sys.exit(1)"""

    output, loss = sess.run([train_step, total_xent], feed_dict=feed_dict)
    total_loss += loss
    report_loss += loss
    report_target_words += target_words
    if (report_target_words - target_words) // report_frequency != report_target_words // report_frequency:
      print >>sys.stderr, 'Parital perp:', math.exp(report_loss / report_target_words)
      report_target_words = 0
      report_loss = 0.0
  print 'Perplexity:', math.exp(total_loss / total_target_words), '(%f loss over %d words)' % (total_loss, total_target_words)
