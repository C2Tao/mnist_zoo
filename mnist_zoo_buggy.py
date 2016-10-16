import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()

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
    t = tf.get_variable(name, initializer = tf_const)
    sess.run(tf.initialize_variables([t]))
    return t

def op_ten_reshape(x, shape):
    y = tf.reshape(x, shape)
    return y

def op_img_pool(x):
    # x: [?, x_width, x_height, x_channel]
    # y: [?, x_width/2, x_height/2, x_channel]
    y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    return y

def op_img_conv(x, n_width, n_height, n_output, name = 'conv'):
    # x: [?, x_width, x_height, n_input]
    # y: [?, x_width/2, x_height/2, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope(name):
        w = var('w', ran(n_width, n_height, n_input, n_output))
        b = var('b', ran(n_output))
        x1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        y = tf.nn.max_pool(x1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    return y

def op_vec_logr(x, n_output, name = 'logr'):
    # x: [?, n_input]
    # y: [?, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope(name):
        w = var('w', ran(n_input, n_output))
        b = var('b', ran(n_output))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
    return y

def op_seq_lstm(x, n_output, name = 'lstm'):
    # x: [?, x_len, n_input]
    # y: [?, x_len, n_output]
    # biggest problem: initial rnn_weights are numerical copies of the given_weight
    #                  cannot symbolically replace rnn_weight with given_weight
    #
    # lstm internal variables [weight, bias]
    # weight is [n_input + n_output, n_output * 4]
    #   stacked like: w = [ w_xi, w_xj, w_xf, w_xo
    #                       w_hi, w_hj, w_hf, w_ho ]
    # bias is [n_output * 4]
    #   stacked like: b = [ b_i, b_j, b_f, b_o ]
    # actual code:
    #   c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * self._activation(j))
    with tf.variable_scope(name) as vs:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_output, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 0.5)
        y, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        
        # manually assign initial value to lstm parameters for customized initialization
        # assign is just a graph operation that assigns value to tensor
        # it does nothing on its own, 
        # need to call sess.run(w_op) after initialization
        # DO NOT CALL intializer_all_variables() again after assign_op
        w_rnn, b_rnn = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        w = var('w', ran(*map(int, w_rnn.get_shape())))
        b = var('b', ran(*map(int, b_rnn.get_shape())))

        init_op = tf.initialize_variables([w_rnn, b_rnn])
        w_op = w_rnn.assign(w)
        b_op = b_rnn.assign(b)

        sess.run([init_op])
        sess.run([w_op, b_op])
    return y

def op_ten_xent(y, y_):
    # y: [?, n_feat]
    # y_: [?, n_feat]
    # e: scalar
    e = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    return e

def print_variable(i, comment = ''):
    variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    print comment
    print map(lambda g: g.name, variables)
    print map(lambda g: g.get_shape(), variables)
    print variables[i].name
    print sess.run(variables[i])

def net_conv(x):
    with tf.variable_scope('conv0'):
        x1 = op_ten_reshape(x, [-1, 28, 28, 1])
        x2 = op_img_conv(x1, 5, 5, 32)
        x3 = op_ten_reshape(x2, [-1, 14 * 14 * 32])
    '''
    with tf.variable_scope('conv1'):
        x = op_ten_reshape(x, [-1, 14, 14, 32])
        x = op_img_conv(x, 5, 5, 64)
        x = op_ten_reshape(x, [-1, 7 * 7 * 64])
    with tf.variable_scope('lstm'):
        x = op_ten_reshape(x, [-1, 7, 7 * 64])
        x = op_seq_lstm(x, 10)
        x = op_ten_reshape(x, [-1, 7 * 10])
    '''
    with tf.variable_scope('out'):
        y = op_vec_logr(x3, 10)
    return y



def net_lstm(x, y_, name):
    with tf.variable_scope(name):
        x = ten_reshape(x, [-1, 28, 28]) # 28 frames of 28 dim feat
        x = seq_lstm(x, 5) # 28 frames of 5 dim feat
        x = ten_reshape(x, [-1, 5 * 28])
        y = vec_logr(x, 10)
        e = ten_xent(y, y_)
    return y, e


x = tf.placeholder(tf.float32, [None,  784])
y_ = tf.placeholder(tf.float32, [None, 10])

y = net_conv(x)
e = op_ten_xent(y, y_)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(e)


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



