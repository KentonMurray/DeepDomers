import sys
import math
import argparse
import tensorflow as tf
import tensorflow.models.rnn.rnn as rnn
import numpy as np
from itertools import izip

tf.set_random_seed(1)

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

def pad(sentence, vocab, length):
  return sentence + max(0, length - len(sentence)) * [vocab.Convert('<pad>')]

parser = argparse.ArgumentParser()
parser.add_argument('source_filename')
parser.add_argument('target_filename')
args = parser.parse_args()

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
max_length = max(len(sent) for sent in source_corpus + target_corpus)
print >>sys.stderr, 'Max length is', max_length

sess = tf.Session()

encoder_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
decoder_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
source_lengths = tf.placeholder(tf.int32, shape=[None])
target_lengths = tf.placeholder(tf.int32, shape=[None])
batch_size = tf.placeholder(tf.int32, shape=[])

fwd_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
rev_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
fwd_encoder = tf.nn.rnn_cell.MultiRNNCell(fwd_encoder_cells * lstm_layer_count)
rev_encoder = tf.nn.rnn_cell.MultiRNNCell(rev_encoder_cells * lstm_layer_count)
decoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
decoder = tf.nn.rnn_cell.MultiRNNCell(decoder_cells * lstm_layer_count)
fwd_initial_state = fwd_encoder.zero_state(batch_size, tf.float32)
rev_initial_state = rev_encoder.zero_state(batch_size, tf.float32)

# Word Embeddings
bos_word_emb = tf.Variable(tf.random_normal([embedding_dim])) # Used when generating target output
source_word_emb = tf.Variable(tf.random_normal([len(source_vocab), embedding_dim], stddev=1.0 / math.sqrt(len(source_vocab))))
target_word_emb = tf.Variable(tf.random_normal([len(target_vocab), embedding_dim], stddev=1.0 / math.sqrt(len(target_vocab))))

# Used to transform the reverse encoder's LSTM state into the decoder's initial state
transform_W = tf.Variable(tf.random_normal([decoder.state_size, 2 * hidden_dim]))
transform_b = tf.Variable(tf.zeros([decoder.state_size]))

# Used to compute alignment/attention
alignment_U = tf.Variable(tf.random_normal([alignment_hidden_dim, 1]))
alignment_W = tf.Variable(tf.random_normal([2 * hidden_dim, alignment_hidden_dim]))
alignment_V = tf.Variable(tf.random_normal([decoder.state_size, alignment_hidden_dim]))
alignment_b = tf.Variable(tf.zeros([alignment_hidden_dim]))

# Used to transform the output of the decoder LSTM into a distribution over target vocabulary
final_W = tf.Variable(tf.random_normal([hidden_dim, len(target_vocab)], stddev=1.0 / math.sqrt(len(target_vocab) + hidden_dim)))
final_b = tf.Variable(tf.zeros([len(target_vocab)]))

def embed_source(source_sentence, source_word_embeddings):
  return tf.nn.embedding_lookup(source_word_embeddings, source_sentence)

input_embs = embed_source(encoder_inputs, source_word_emb)
input_embs_list = tf.unpack(tf.transpose(input_embs, perm=[1, 0, 2]))
annotations, fwd_state, rev_state = rnn.bidirectional_rnn(fwd_encoder, rev_encoder, input_embs_list, fwd_initial_state, rev_initial_state, sequence_length=source_lengths)
decoder_initial_state = tf.matmul(rev_state, transform_W) + transform_b

state = decoder_initial_state
prev_word = tf.reshape(tf.tile(tf.expand_dims(bos_word_emb, 0), tf.pack([batch_size, 1])), [-1, embedding_dim])
annotation_matrix = tf.pack(annotations) # (max_length, ?, 2*hidden_dim)

# annotation_matrix, hidden_dim, state, alignment_V, alignment_b, alignment_W
# max_length, alignment_hidden_dim alignment_U
def compute_attention(output_state, annotation_matrix, alignment_U, alignment_V, alignment_W, alignment_b, hidden_dim, alignment_hidden_dim, max_length):
  # The attention vector A should have dimensions (?, max_length)
  # and is generated from the annotation matrix I (max_length, ?, 2*hidden_dim) => (max_length * ?, 2*hidden_dim)
  # and the output state S (?, hidden_dim).
  # We want A = UH, H = tanh(WI + VS + b)
  # Where H is (?, max_length, alignment_hidden_dim)

  # \Alpha is the normalized version of A: \Alpha = softmax(A)
  annotation_matrix_r = tf.reshape(annotation_matrix, [-1, 2 * hidden_dim]) # (?*max_length, 2*hidden_dim)
  bias_term = tf.matmul(state, alignment_V) + alignment_b # [?, alignment_hidden_dim], will be broadcast to [S, ?, alignment_hidden_dim]
  IW = tf.matmul(annotation_matrix_r, alignment_W)
  IW_r = tf.reshape(IW, [max_length, -1, alignment_hidden_dim])
  alignment_hidden = tf.nn.tanh(IW_r + bias_term) # [S, ?, alignment_hidden_dim]
  H_r = tf.reshape(alignment_hidden, [-1, alignment_hidden_dim])
  alignment = tf.matmul(H_r, alignment_U)
  attention = tf.nn.softmax(alignment) # [S * ?, 1]
  attention_r = tf.reshape(attention, [max_length, -1, 1]) # [S, ?, 1]
  return attention_r

def compute_context(annotation_matrix, attention):
  # We essentially want I^T * A, but the batch sizes get in the way.
  # We solve this by changing the dot product into two steps:
  # an element-wise product, and a summation.
  IA = tf.mul(annotation_matrix, attention)
  return tf.reduce_sum(IA, 0) # [?, 2H]

decoder_input_embs = tf.nn.embedding_lookup(target_word_emb, decoder_inputs)
# This is basically just an unrolled version of dynamic_rnn, which gives us access to te
# states and outputs one at a time, so we can compute attention and whatnot.
input_embs_t = tf.transpose(decoder_input_embs, perm=[1, 0, 2])
outputs = []
for t, input_emb in enumerate(tf.unpack(input_embs_t)):
  if t > 0:
    tf.get_variable_scope().reuse_variables()

  attention = compute_attention(state, annotation_matrix, alignment_U, alignment_V, alignment_W, alignment_b, hidden_dim, alignment_hidden_dim, max_length)
  context = compute_context(annotation_matrix, attention)

  prev_word_and_context = tf.concat(1, [tf.reshape(prev_word, [-1, embedding_dim]), context]) 
 
  output, state = decoder(prev_word_and_context, state) # Output should be (?, H)
  outputs.append(output)
  prev_word = input_emb
decoder_outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2]) # (?, T, H)

# We want to multiply this tensor by final_W, which is (hidden_dim, target_vocab_size).
# This is equivalent to doing a matrix multiply (?*max_length, hidden_dim) times (hidden_dim, target_vocab_size)
# Since TF doesn't let us do tensor-matrix products, we use this transformation as a hackaround.
decoder_outputs_t = tf.transpose(decoder_outputs, perm=[1, 0, 2]) # (max_length, ?, hidden_dim)
decoder_outputs_tr = tf.reshape(decoder_outputs_t, [-1, hidden_dim]) # (max_length * ?, hidden_dim)
output_dists = tf.matmul(decoder_outputs_tr, final_W) + final_b # (max_length * ?, target_vocab_size)

# The distributions are now in one big matrix. We could reshape it, back to a 3-tensor, but
# instead we just reshape the references from a (?, max_length) matrix into a (? * max_length) vector
# and compute the loss with that.
decoder_inputs_r = tf.reshape(decoder_inputs, [-1])
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output_dists, decoder_inputs_r)
total_xent = tf.reduce_sum(cross_entropy)

#optimizer = tf.train.GradientDescentOptimizer(0.1)
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(total_xent)
sess.run(tf.initialize_all_variables())

total_target_words = sum(len(sent) for sent in target_corpus)
for i in range(100000):
  report_loss = 0.0
  total_loss = 0.0
  report_target_words = 0
  for j in range(0, len(source_corpus), minibatch_size):
    source_inputs = []
    target_inputs = []
    src_lengths = []
    tgt_lengths = []
    J = min(j + minibatch_size, len(source_corpus))
    N = max(len(source_corpus[k]) for k in range(j, J))
    M = max(len(target_corpus[k]) for k in range(j, J))
    for k in range(j, J):
      padded_source = pad(source_corpus[k], source_vocab, max_length)
      padded_target = pad(target_corpus[k], target_vocab, max_length)
      source_inputs.append(padded_source)
      target_inputs.append(padded_target)
      src_lengths.append(len(source_corpus[k]))
      tgt_lengths.append(len(target_corpus[k]))
    feed_dict = {}
    feed_dict[batch_size.name] = len(source_inputs)
    feed_dict[source_lengths.name] = np.array(src_lengths)
    feed_dict[target_lengths.name] = np.array(tgt_lengths)
    feed_dict[encoder_inputs.name] = np.array(source_inputs)
    feed_dict[decoder_inputs.name] = np.array(target_inputs) 

    output, loss = sess.run([train_step, total_xent], feed_dict=feed_dict)
    total_loss += loss
    report_loss += loss
    report_target_words += sum(tgt_lengths)
    if (report_target_words - sum(tgt_lengths)) // 1000 != report_target_words // 1000:
      print >>sys.stderr, 'Parital perp:', math.exp(report_loss / report_target_words)
      report_target_words = 0
      report_loss = 0.0
  print 'Perplexity:', math.exp(total_loss / total_target_words), '(%f loss over %d words)' % (total_loss, total_target_words)
