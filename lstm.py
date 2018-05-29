import tensorflow as tf
import numpy as np


'''
Let:
  - c be of shape (m, d_c)
  - h be of shape (m, d_h)
  - x be of shape (m, d_x)
'''

d_c = 15
d_h = d_c
d_x = 27

m = 2
T = 5

def batch_mm(unbatched, batched):
    return tf.einsum('ij,lj->li', unbatched, batched)

'''
Produces one LSTM recurrence.
'''
def lstm_unit(c, h, x, params):
    W_f, b_f = params['W_f'], params['b_f']
    W_i, b_i = params['W_i'], params['b_i']
    W_c, b_c = params['W_c'], params['b_c']
    W_o, b_o = params['W_o'], params['b_o']

    hx = tf.concat([h, x], axis=1, name='input_concat')
    f = tf.sigmoid(batch_mm(W_f, hx) + b_f)
    i = tf.sigmoid(batch_mm(W_i, hx) + b_i)
    o = tf.sigmoid(batch_mm(W_o, hx) + b_o)
    c_tilde = tf.tanh(batch_mm(W_c, hx) + b_c)
    c_tilde *= i
    c *= f
    c += c_tilde
    o *= tf.tanh(c)
    return c, o


params = {
    'W_f': tf.get_variable(name='W_f', shape=(d_c, d_h + d_x)),
    'W_i': tf.get_variable(name='W_i', shape=(d_c, d_h + d_x)),
    'W_c': tf.get_variable(name='W_c', shape=(d_c, d_h + d_x)),
    'W_o': tf.get_variable(name='W_o', shape=(d_c, d_h + d_x)),
    'b_f': tf.get_variable(name='b_f', shape=(d_c)),
    'b_i': tf.get_variable(name='b_i', shape=(d_c)),
    'b_c': tf.get_variable(name='b_c', shape=(d_c)),
    'b_o': tf.get_variable(name='b_o', shape=(d_c))
}

c_0 = tf.zeros((m, d_c))
h_0 = tf.zeros((m, d_h))

x = tf.placeholder(tf.float32, (T, m, d_x))

c_t = c_0
h_t = h_0
for t in range(T):
    c_t, h_t = lstm_unit(c_t, h_t, x[t], params)

import random

def generate_example():
    base = np.zeros((T, m, d_x))
    tot = np.zeros((m))
    for t in range(T):
        for i in range(m):
            digit = random.randint(0, 9)
            tot[i] += digit * 10 ** t
            base[t, i, digit] = 1
    return base, tot

truth = tf.placeholder(tf.float32, (m))

# predictions = tf.layers.dense(c_t, 1, activation=tf.nn.relu)
# predictions = tf.squeeze(predictions, axis=-1)
predictions = c_t[:, 0] * 10 ** (T + 1)

loss = tf.squared_difference(predictions, truth)
loss = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(1e-2)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        X, y = generate_example()
        c, _ = sess.run([loss, train_op], feed_dict={x: X, truth: y})
        if i % 1000 == 0:
            p, t = sess.run([predictions, truth], feed_dict={x: X, truth: y})
            print(p, t, c)
