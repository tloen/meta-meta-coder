import tensorflow as tf
import numpy as np
from lstm import lstm_unit

dim_in = 20
dim_main = 100
dim_out = 20
dim_x = 10
m = 1

pr = 4

T = 5

def get_params(d, d_x, suffix):
    return {
        'W_f': tf.get_variable(name='W_f' + suffix, shape=(d, d + d_x)),
        'W_i': tf.get_variable(name='W_i' + suffix, shape=(d, d + d_x)),
        'W_c': tf.get_variable(name='W_c' + suffix, shape=(d, d + d_x)),
        'W_o': tf.get_variable(name='W_o' + suffix, shape=(d, d + d_x)),
        'b_f': tf.get_variable(name='b_f' + suffix, shape=(d)),
        'b_i': tf.get_variable(name='b_i' + suffix, shape=(d)),
        'b_c': tf.get_variable(name='b_c' + suffix, shape=(d)),
        'b_o': tf.get_variable(name='b_o' + suffix, shape=(d))
    }

def param_condition_stitch(index, param_vers):
    if len(param_vers) == 1:
        return param_vers[0]
    return tf.cond(
      tf.equal(index, 0), 
      true_fn=lambda: param_vers[0],
      false_fn=lambda: param_condition_stitch(index - 1, param_vers[1:])
    )

param_names = ['W_f', 'W_i', 'W_c', 'W_o', 'b_f', 'b_i', 'b_c', 'b_o']
def params_condition_stitch(index, param_vers):
    return dict([
        (param_name, param_condition_stitch(index, [
            param_ver[param_name] for param_ver in param_vers
        ]))
        for i, param_name in enumerate(param_names)
    ])

def meta_lstm_unit(cin, hin, cmain, hmain, cout, hout, x, params_in, params_main, params_out):
    cin, hin = lstm_unit(cin, hin, x, params_in)
    cmain, hmain = lstm_unit(cmain, hmain, hin, params_main)
    cout, hout = lstm_unit(cout, hout, hmain, params_out)
    return cin, hin, cmain, hmain, cout, hout

cin = cin_0 = tf.zeros((m, dim_in))
hin = hin_0 = tf.zeros((m, dim_in))
cmain = cmain_0 = tf.zeros((m, dim_main))
hmain = hmain_0 = tf.zeros((m, dim_main))
cout = cout_0 = tf.zeros((m, dim_out))
hout = hout_0 = tf.zeros((m, dim_out))

problem_index = tf.placeholder(tf.int32, [])

paramsets_in = [get_params(dim_in, dim_x, '_in_%d' % i) for i in range(pr)]
params_in = params_condition_stitch(problem_index, paramsets_in)

#params_in = get_params(dim_in, dim_x, '_in')
params_main = get_params(dim_main, dim_in, '_main')

paramsets_out = [get_params(dim_out, dim_main, '_out_%d' % i) for i in range(pr)]
params_out = params_condition_stitch(problem_index, paramsets_out)
#params_out = get_params(dim_out, dim_main, '_out')

x = tf.placeholder(tf.float32, (T, m, dim_x))

outputs = []

for t in range(T):
    cin, hin, cmain, hmain, cout, hout = meta_lstm_unit(cin, hin, cmain, hmain, cout, hout, x[t], params_in, params_main, params_out)
    outputs.append(hout)

import random

def generate_example():
    base = np.zeros((T, m, dim_x))
    tot = np.zeros((m))
    for t in range(T):
        for i in range(m):
            digit = random.randint(0, 9)
            tot[i] += digit * 10 ** t
            base[t, i, digit] = 1
    return base, tot

truth = tf.placeholder(tf.float32, (m))

predictions = outputs[-1][:, 0] * 10 ** (T + 1)

loss = tf.squared_difference(predictions, truth)
loss = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(1e-2)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        X, y = generate_example()
        c, _ = sess.run([loss, train_op], feed_dict={x: X, truth: y, problem_index: 0})
        if i % 1000 == 0:
            p, t = sess.run([predictions, truth], feed_dict={x: X, truth: y, problem_index: 0})
            print(p, t, c)

    
