List of TensorFlow Tips and Tricks
==================================

Tensor-matrix (or tensor-tensor) multiplication
-----------------------------------------------

Tensorflow does not support multiplication of tensors of rank >2. We can hack around this in the following way.
Suppose we have a tensor A with dimensions [X, Y, Z] and a tensor B with dimensions [W, X] (or [V, W, X]) that
we wish to multiply together as BA, resulting in a [W, Y, Z] (or [V, W, Y, Z]) tensor. We can achieve this by first
reshaping to combine all the axes that will not be multiplied together:
```python
A_r = tf.reshape(A, [X, Y*Z])
# B_r = tf.reshape(B, [V * W, X]) # In the tensor-tensor case
```

Now we can matrix multiply as normal:
```python
intermediate = tf.matmul(B, A_r)
```

and then unpack the result into the dimensions we actually want.
```python
result = tf.reshape(intermediate, [W, Y, Z])
```

Note that this does work if one of the dimensions is the batch size variable (represented as -1 in the dimensions).
For example, if A has dimensions [-1, X, Y] and B has [Y, Z], and we want to get out a [-1, X, Z] tensor, we would do:

```python
A_r = tf.reshape(A, [-1, Y]) # Variable dimension times anything is still variable. I.e. -1 * X = -1, by TensorFlow dimension logic.
intermediate = tf.matmul(A_r, B) # Result will show as [-1, Z] but is really [-1 * X, Z]
result = tf.reshape(intermediate, [-1, X, Z])
```

Matrix multiplication with batches
----------------------------------

Suppose we have a matrix with dimensions [H, S] and a vector with shape [S] that we wish to multipy together to get a vector
of size [H]. This is easy without batches, but with batches we instead have [?, H, S] and [?, S], and desire an output of shape [?, H].

Multiplying these dimensions together cannot ever give [?, H].
If we multiply along the batch dimension, the output will have no batch size at all.
If we don't, then we'll be left with *two* batch dimensions.

The solution is to break the dot product into two steps: an element-wise product and a summation. Let M be [?, H, S] and V be [?, S].

```python
V_e = tf.expand_dims(V, 1) # Now V_e is [?, 1, S]
intermediate = tf.mul(M, V) # Element-wise multiplication. V_e will be broadcost along the "H" dimension, giving us a [?, H, S] tensor.
result = tf.reduce_sum(intermediate, 1) # Sum along the 1st (0-indexed) column. This is now [?, H]
```
