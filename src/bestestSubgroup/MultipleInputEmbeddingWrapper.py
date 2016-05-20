from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
import tensorflow as tf

import math

class MultipleInputEmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes, embedding_sizes, initializer=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer list specifying vocabulary sizes
      embedding_sizes: integer list specifying the dimensions of the embeddings
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    for embedding_class in embedding_classes:
        if embedding_class <= 0:
            raise ValueError("Embedding_class must be > 0: "
                       "%d." % (embedding_class))
    if type(embedding_sizes) == int:
        embedding_sizes = [embedding_sizes] * len(embedding_classes)

    for embedding_size in embedding_sizes:
        if embedding_size <= 0:
            raise ValueError("Embedding_size must be > 0: "
                       "%d." % (embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_sizes = embedding_sizes
    self._initializer = initializer

  @property
  def input_size(self):
    return 1

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with vs.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper2"
      with ops.device("/cpu:0"):
        if self._initializer:
          initializer = self._initializer
        elif vs.get_variable_scope().initializer:
          initializer = vs.get_variable_scope().initializer
        else:
          # Default initializer for embeddings should have variance=1.
          sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
          initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
        embeddings = []
        for i in xrange(len(self._embedding_classes)):
            embeddings.append(vs.get_variable("embedding"+str(i), [self._embedding_classes[i],
                                                  self._embedding_sizes[i]],
                                    initializer=initializer))
        embedded = []
        for i in xrange(len(self._embedding_classes)):
            embedded.append(embedding_ops.embedding_lookup(
                  embeddings[i], array_ops.reshape(inputs[i], [-1])))

        finalEmbedded = tf.concat(1, embedded)

    return self._cell(finalEmbedded, state)