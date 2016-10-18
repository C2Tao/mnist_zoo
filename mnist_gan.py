import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
graph = tf.Graph()
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

def zero(*shape):
    return np.zeros(shape, dtype = np.float32)

def var(name, np_const):
    tf_const = tf.cast(tf.constant(np_const), tf.float32)
    t = tf.get_variable(name, initializer = tf_const)
    sess.run(tf.initialize_variables([t]))
    return t

def ten_reshape(x, shape):
    y = tf.reshape(x, shape)
    return y

def img_pool(x):
    # x: [?, x_width, x_height, x_channel]
    # y: [?, x_width/2, x_height/2, x_channel]
    y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    return y

def img_conv(x, n_width, n_height, n_output):
    # x: [?, x_width, x_height, n_input]
    # y: [?, x_width, x_height, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope('conv'):
        w = var('w', zero(n_width, n_height, n_input, n_output))
        b = var('b', zero(n_output))
        y = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    return y

def img_deconv(x, n_width, n_height, n_output):
    n_input = x.get_shape()[-1]
    with tf.name_scope('deconv') as scope:
        deconv = tf.nn.conv2d_transpose(input_layer, [3, 3, 1, 1], 
             [1, 26, 20, 1], [1, 2, 2, 1], padding='SAME', name=None)
    return y

def vec_sigm(x, n_output):
    # x: [?, n_input]
    # y: [?, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope('logr'):
        w = var('w', ran(n_input, n_output)*0.001)
        b = var('b', ran(n_output)*0.001)
        y = tf.nn.sigmoid(tf.matmul(x, w) + b)
    return y
def get_var(scope):
    return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

def vec_logr(x, n_output):
    # x: [?, n_input]
    # y: [?, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope('logr'):
        w = var('w', ran(n_input, n_output))
        b = var('b', ran(n_output))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
    return y

def seq_lstm(x, n_output):
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
    with tf.variable_scope('lstm') as vs:
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

def ten_xent(y, y_):
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

def net_conv(x, y_, name):
    with tf.variable_scope(name):
        x = ten_reshape(x, [-1, 28, 28, 1])
        x = img_conv(x, 5, 5, 32)
        x = img_pool(x)
        x = ten_reshape(x, [-1, 14 * 14 * 32])
        y = vec_logr(x, 10)
        e = ten_xent(y, y_)
    return y, e

def net_logr(x,  y_, name):
    with tf.variable_scope(name):
        y = vec_logr(x, 10)
        e = ten_xent(y, y_)
    return y, e


def net_lstm(x, y_, name):
    with tf.variable_scope(name):
        x = ten_reshape(x, [-1, 28, 28]) # 28 frames of 28 dim feat
        x = seq_lstm(x, 5) # 28 frames of 5 dim feat
        x = ten_reshape(x, [-1, 5 * 28])
        y = vec_logr(x, 10)
        e = ten_xent(y, y_)
    return y, e

def plot_img(x):
    matrix = np.reshape(x, [28, 28])
    import matplotlib.pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()
'''
def generator(c):
    # input sample from random noise
    # output counterfiet example
    with tf.variable_scope('generator')
        x_gen = vec_sigm(p, 784)
    return x
def discriminator(x):
    # input sample to be judged
    # output probability p for passing the discriminator test
    with tf.variable_scope('discriminator') as scope:
        p = vec_sigm(x, 1)
        scope.reuse_variables()
    return p
'''
n_code = 100
x_sam = tf.placeholder(tf.float32, [None,  784])
y = tf.placeholder(tf.float32, [None, 10])
p = tf.placeholder(tf.float32, [None, n_code])

with tf.variable_scope('generator'):
    x_gen = vec_sigm(p, 784)

with tf.variable_scope('discriminator') as scope:
    # p stands for passing the discriminator test
    p_gen = vec_sigm(x_gen, 1)
    scope.reuse_variables()
    p_sam = vec_sigm(x_sam, 1)

obj_gen = tf.log(1.0-p_gen) #minimize
obj_sam = tf.log(p_sam) + tf.log(1.0-p_gen) #maximize

gen_step = tf.train.GradientDescentOptimizer(0.001).minimize(obj_gen, var_list=get_var('generator'))
dis_step = tf.train.GradientDescentOptimizer(0.001).minimize(-obj_sam, var_list = get_var('discriminator'))

#print_variable(1, 'before training')
#print_variable(3, 'before training')

#x_img = sess.run(x_gen, feed_dict={p: zero(1, 5)})

#plot_img(sess.run(x_gen, feed_dict={p: zero(1, 5)}))
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_ps = ran(100, n_code)
    sess.run(dis_step, feed_dict={x_sam: batch_xs, p: batch_ps})
    sess.run(gen_step, feed_dict={x_sam: batch_xs, p: batch_ps})
    #print 'loss', np.exp(sess.run(obj_gen, feed_dict={x_sam: batch_xs, p: batch_ps}))
    

#x_img, _ = mnist.train.next_batch(1)
plot_img(sess.run(x_gen, feed_dict={p: zero(1, n_code)}))


#print_variable(1, 'after training')
#print_variable(3, 'after training')

#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



