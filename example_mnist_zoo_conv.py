import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

np.random.seed(0)
tf.set_random_seed(0)

def test_rand_example():
    #using tf
    #tf.set_random_seed(0)
    initial = tf.truncated_normal([5,5], stddev=0.1)
    print sess.run(initial)

    #using np
    #np.random.seed(0)
    init = tf.constant(np.random.rand(1, 2))
    x = tf.get_variable('var_name', initializer=init)
    sess.run(tf.initialize_all_variables())
    print sess.run(x)

def ran(*shape):
    return np.random.rand(*shape)

def var(name, np_const):
    tf_const = tf.cast(tf.constant(np_const), tf.float32)
    return tf.get_variable(name, initializer = tf_const)

def layer_reshape(x, shape):
    y = tf.reshape(x, shape)
    return y

def layer_pool(x):
    y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    return y

def layer_conv(x, n_width, n_height, n_output):
    n_input = x.get_shape()[-1]
    with tf.variable_scope('conv'):
        w = var('w', ran(n_width, n_height, n_input, n_output))
        b = var('b', ran(n_output))
        y = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    return y

def layer_logr(x, n_output):
    n_input = x.get_shape()[-1]
    with tf.variable_scope('logr'):
        w = var('w', ran(n_input, n_output))
        b = var('b', ran(n_output))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
    return y

def layer_xent(y, y_):
    e = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    return e

x = tf.placeholder(tf.float32, [None,  784])
y_ = tf.placeholder(tf.float32, [None, 10])

def net_conv(x, y_, name):
    with tf.variable_scope(name):
        x1 = layer_reshape(x, [-1, 28, 28, 1])
        x2 = layer_conv(x1, 5, 5, 32)
        x3 = layer_pool(x2)
        x4 = layer_reshape(x3, [-1, 14 * 14 * 32])
        y = layer_logr(x4, 10)
        e = layer_xent(y, y_)
    return y, e

def net_logr(x,  y_, name):
    with tf.variable_scope(name):
        y = layer_logr(x, 10)
        e = layer_xent(y, y_)
    return y, e

'''
def net_lstm(x, y_, name):
    with tf.variable_scope(name):
        x1 = layer_reshape(x, [-1, 28, 28, 1])
        x4 = layer_reshape(x3, [-1, 14 * 14 * 32])
        y = layer_logr(x, 10)
        e = layer_xent(y, y_)
    return y, e
''' 


y, e = net_conv(x, y_, 'conv')
#y, e = net_logr(x, y_, 'logr')


#cross_ent = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(e)

sess.run(tf.initialize_all_variables())
for i in range(10):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



