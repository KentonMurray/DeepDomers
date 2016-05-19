#!/usr/bin/env python

import sys
import math
import argparse
import tensorflow as tf
import tensorflow.models.rnn.rnn as rnn
import numpy as np
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
#print("source_corpus=%s target_corpus=%s" % (len(source_corpus), len(target_corpus)) )
assert len(source_corpus) == len(target_corpus)
print >>sys.stderr, 'Vocab sizes: %d/%d' % (len(source_vocab), len(target_vocab))

minibatch_size = 7
embedding_dim = 16
hidden_dim = 32
lstm_layer_count = 1
max_length = max(len(sent) for sent in source_corpus + target_corpus)
print >>sys.stderr, 'Max length is', max_length

sess = tf.Session()

source_word_emb = tf.Variable(tf.random_normal([len(source_vocab), embedding_dim], stddev=1.0 / math.sqrt(len(source_vocab))))
target_word_emb = tf.Variable(tf.random_normal([len(target_vocab), embedding_dim], stddev=1.0 / math.sqrt(len(target_vocab))))

final_W = tf.Variable(tf.random_normal([hidden_dim, len(target_vocab)], stddev=1.0 / math.sqrt(len(target_vocab) + hidden_dim)))
final_b = tf.Variable(tf.zeros([len(target_vocab)]))

fwd_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
rev_encoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
fwd_encoder = tf.nn.rnn_cell.MultiRNNCell(fwd_encoder_cells * lstm_layer_count)
rev_encoder = tf.nn.rnn_cell.MultiRNNCell(rev_encoder_cells * lstm_layer_count)
decoder_cells = [tf.nn.rnn_cell.BasicLSTMCell(hidden_dim) for i in range(lstm_layer_count)]
decoder = tf.nn.rnn_cell.MultiRNNCell(decoder_cells * lstm_layer_count)
batch_size = tf.placeholder(tf.int32, shape=[])
fwd_initial_state = fwd_encoder.zero_state(batch_size, tf.float32)
rev_initial_state = rev_encoder.zero_state(batch_size, tf.float32)

encoder_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
decoder_inputs = tf.placeholder(tf.int32, shape=[None, max_length])
source_lengths = tf.placeholder(tf.int32, shape=[None])
target_lengths = tf.placeholder(tf.int32, shape=[None])

with tf.variable_scope('fwd_encoder'):
  input_embs = tf.nn.embedding_lookup(source_word_emb, encoder_inputs) # (?, max_length, embedding_dim)
  fwd_outputs, state = rnn.dynamic_rnn(fwd_encoder, input_embs, initial_state = fwd_initial_state, sequence_length=source_lengths)
with tf.variable_scope('rev_encoder'):
  input_embs = tf.reverse(input_embs, [False, True, False]) 
  rev_outputs_reversed, state = rnn.dynamic_rnn(rev_encoder, input_embs, initial_state = rev_initial_state, sequence_length=source_lengths)
  rev_outputs = tf.reverse(rev_outputs_reversed, [False, True, False])

outputs = tf.concat(2, [fwd_outputs, rev_outputs])

transform_W = tf.Variable(tf.random_normal([2 * hidden_dim, decoder.state_size]))
transform_b = tf.Variable(tf.zeros([decoder.state_size]))
context = tf.concat(1, [fwd_outputs[:, max_length - 1, :], rev_outputs[:, 0, :]])
decoder_initial_state = tf.matmul(context, transform_W) + transform_b

with tf.variable_scope('decoder'):
  input_embs = tf.nn.embedding_lookup(target_word_emb, decoder_inputs)
  outputs2, states2 = rnn.dynamic_rnn(decoder, input_embs, initial_state = decoder_initial_state, sequence_length=target_lengths)

# We want to multiply this tensor by final_W, which is (hidden_dim, target_vocab_size).
# This is equivalent to doing a matrix multiply (?*max_length, hidden_dim) times (hidden_dim, target_vocab_size)
# Since TF doesn't let us do tensor-matrix products, we use this transformation as a hackaround.
relevant_outputs_t = tf.transpose(outputs2, perm=[1, 0, 2]) # (max_length, ?, hidden_dim)
relevant_outputs_tr = tf.reshape(relevant_outputs_t, [-1, hidden_dim]) # (max_length * ?, hidden_dim)
relevant_dists = tf.matmul(relevant_outputs_tr, final_W) + final_b # (max_length * ?, target_vocab_size)

# The distributions are now in one big matrix. We could reshape it, back to a 3-tensor, but
# instead we just reshape the references from a (?, max_length) matrix into a (? * max_length) vector
# and compute the loss with that.
decoder_inputs_r = tf.reshape(decoder_inputs, [-1])
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(relevant_dists, decoder_inputs_r)
average_xent = tf.reduce_mean(cross_entropy)

#optimizer = tf.train.GradientDescentOptimizer()
optimizer = tf.train.AdamOptimizer(1.0)
train_step = optimizer.minimize(average_xent)
sess.run(tf.initialize_all_variables())

total_target_words = sum(len(sent) for sent in target_corpus)
for i in range(10000):
  total_loss = 0.0
  for j in range(0, len(source_corpus), minibatch_size):
    source_inputs = []
    target_inputs = []
    src_lengths = []
    tgt_lengths = []
    for k in range(j, min(j + minibatch_size, len(source_corpus))):
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
    output, loss = sess.run([train_step, average_xent], feed_dict=feed_dict)
    total_loss += loss
  print 'Perplexity:', math.exp(total_loss / total_target_words)
