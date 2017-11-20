# This file is part of /tensorflow-mnist-example

import tensorflow as tf
import numpy as np

# tf constants
hello = tf.constant('hello world')
print(type(hello))  # Tensor object

x = tf.constant(100)
print(x)

# Create a tf session
sess = tf.Session()

print(sess.run(hello))
print(type(sess.run(hello)))

# tf operations
x = tf.constant(2)
y = tf.constant(3)

# How to perform operations on tf.constant()
with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition : ', sess.run(x + y))
    print('Subtraction : ', sess.run(x - y))
    print('Multiplication : ', sess.run(x * y))
    print('Division : ', sess.run(x / y))

# Place holders : helpful to define tf functions

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x, y)
sub = tf.subtract(x, y)
multi = tf.multiply(x, y)

with tf.Session() as sess:
    print('Operations with Placeholders')
    print('Addition : ', sess.run(add, feed_dict={x: 20, y: 30}))  # We pass a dict for the inputs
    print('Subtraction : ', sess.run(sub, feed_dict={x: 20, y: 30}))
    print('Multiplication : ', sess.run(multi, feed_dict={x: 20, y: 30}))

# Matrix multiplication example

a = np.array([[5.0, 5.0]])
b = np.array(([[2.0], [2.0]]))

mat1 = tf.constant(a)
mat2 = tf.constant(b)

matrix_multi = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print(result)
