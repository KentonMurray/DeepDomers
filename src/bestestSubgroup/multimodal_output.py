# This is to see if we can have multiple inputs and outputs

import math
import tensorflow as tf
import numpy as np


## Build a graph.
#a = tf.constant(5.0)
#b = tf.constant(6.0)
#c = a * b
#d = b * b
#
## Launch the graph in a session.
#sess = tf.Session()
#
## Evaluate the tensor `c`.
#print(sess.run(c))
#print(sess.run(d))


HIDDEN_NODES = 10

x = tf.placeholder(tf.float32, [None, 2])
W_hidden = tf.Variable(tf.truncated_normal([2, HIDDEN_NODES], stddev=1./math.sqrt(2)))
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]))
hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

W_logits = tf.Variable(tf.truncated_normal([HIDDEN_NODES, 2], stddev=1./math.sqrt(HIDDEN_NODES)))
b_logits = tf.Variable(tf.zeros([2]))
logits = tf.matmul(hidden, W_logits) + b_logits

y = tf.nn.softmax(logits)
z_logits = tf.matmul(hidden, W_logits) - b_logits

y_input = tf.placeholder(tf.float32, [None, 2])
z_input = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_input)
#loss = tf.reduce_mean(cross_entropy)
y_loss = tf.reduce_mean(cross_entropy)

#train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

z_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(z_logits, z_input)
z_loss = tf.reduce_mean(z_cross_entropy)

#z_train_op = tf.train.GradientDescentOptimizer(0.2).minimize(z_loss)
loss = y_loss + z_loss
train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yTrain = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
zTrain = np.array([[0.9, 0], [0, 0.9], [0, 0.9], [0.9, 0]])

for i in xrange(500):
  #_, loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain})
  #_, z_loss_val = sess.run([z_train_op, z_loss], feed_dict={x: xTrain, z_input: zTrain})
  _, loss_val = sess.run([train_op, loss], feed_dict={x: xTrain, y_input: yTrain, z_input: zTrain})

  if i % 10 == 0:
    #print "Step:", i, "Current loss:", loss_val, "z_loss:", z_loss_val
    print "Step:", i, "Current loss:", loss_val
