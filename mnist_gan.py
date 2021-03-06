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
        w = var('w', ran(n_width, n_height, n_input, n_output)*0.001)
        b = var('b', zero(n_output))
        y = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    return y

def img_full(x, n_hidden, name = 'cnn', n_width = 5, n_height = 5):
    # x: [?, x_width, x_height, n_input]
    # y: [?, x_width/2^L, x_height/2^L, n_hidden[L-1]], L = |n_hidden|
    n_input = x.get_shape()[-1]
    for i, n_output in enumerate(n_hidden):
        with tf.variable_scope(name + str(i)):
            w = var('w', ran(n_width, n_height, n_input, n_output)*0.001)
            b = var('b', zero(n_output))
            x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
            n_input = n_output
    return x

def img_deconv(x, n_width, n_height, n_output):
    n_input = x.get_shape()[-1]
    with tf.name_scope('deconv') as scope:
        y = tf.nn.conv2d_transpose(x, w, [1, 1, 1, 1], [1, 2, 2, 1], padding='SAME')
    return y

def vec_full(x, n_hidden, name = 'full', activation = tf.nn.relu):
    # x: [?, n_input]
    # y: [?, n_hidden[-1]]
    n_input = x.get_shape()[-1]
    for i, n_output in enumerate(n_hidden):
        with tf.variable_scope(name + str(i)):
            w = var('w', ran(n_input, n_output)*0.001)
            b = var('b', zero(n_output))
            x = activation(tf.matmul(x, w) + b)
            n_input = n_output
    return x

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
    if i>0:
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
    matrix = np.reshape(x, [-1, 28])
    import matplotlib.pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()

n_code = 100
n_iter = 10000
n_batch = 100
lr_rate = 0.001

def generator_attention(c):
    # input sample from random noise
    # output counterfiet example
    x1 = vec_full(c, [30, 784], activation = tf.nn.sigmoid, name = 'l1')
    x2 = vec_full(c, [30, 784], activation = tf.nn.sigmoid, name = 'l2')
    x3 = vec_full(c, [30, 784], activation = tf.nn.sigmoid, name = 'l3')
    xs = tf.concat(1, [x1,x2,x3])
    xr = tf.reshape(xs, [-1, 3, 784])
    
    a0 = vec_full(c, [30, 3 * 784], activation = tf.nn.sigmoid, name = 'a0')
    a1 = tf.reshape(xs, [-1, 3, 784])
    ar = tf.nn.softmax(a1, dim = 1)
    
    x = tf.reduce_sum(xr*ar, reduction_indices = 1)
    #x = (x1+x2+x3)/3
    return x, ar, x1,x2,x3

def generator(c):
    # input sample from random noise
    # output counterfiet example
    c = vec_full(c, [100], activation = tf.nn.relu, name = 'relu')
    x = vec_full(c, [784], activation = tf.nn.sigmoid, name = 'sig')
    return x

def discriminator(x):
    # input sample to be judged
    # output probability p for passing the discriminator test

    #x = ten_reshape(x, [-1, 28, 28, 1])
    #x = img_full(x, [4, 16])
    #x = ten_reshape(x, [-1, 7 * 7 * 64])
    #p = vec_full(x, [100, 1])
    x = vec_full(x, [100], activation = tf.nn.relu, name = 'relu')
    p = vec_full(x, [1], activation = tf.nn.sigmoid, name = 'sig')
    return p

x_sam = tf.placeholder(tf.float32, [None,  784])
y = tf.placeholder(tf.float32, [None, 10])
c = tf.placeholder(tf.float32, [None, n_code])

with tf.variable_scope('generator'):
    x_gen = generator(c)
print x_gen.name

with tf.variable_scope('discriminator') as scope:
    p_sam = discriminator(x_sam)
    scope.reuse_variables()
    p_gen = discriminator(x_gen)

print p_gen.name
print p_sam.name


def mat_std(x):
    #input  [n_sample, n_feat]
    #output [n_sample] of standard deviation
    x_u = tf.reduce_mean(x, reduction_indices = 0)
    x_n = tf.reduce_mean((x-x_u)**2,reduction_indices = 0)
    return tf.sqrt(x_n)
    
    

#s_gen = mat_std(tf.reshape(x_gen, [-1, 784]))
#s_sam = mat_std(tf.reshape(x_sam, [-1, 784]))
#obj_std = - tf.reduce_mean((s_gen-s_sam)**2)
#w_std = 1

obj_gen_early = -tf.log(p_gen)  #+ w_std * obj_std
obj_gen = tf.log(1.0-p_gen) #+ w_std * obj_std 
obj_sam = - tf.log(p_sam) - tf.log(1.0-p_gen)

gen_step_early = tf.train.AdagradOptimizer(lr_rate).minimize(obj_gen_early, var_list=get_var('generator'))
gen_step = tf.train.AdagradOptimizer(lr_rate).minimize(obj_gen, var_list=get_var('generator'))
dis_step = tf.train.AdagradOptimizer(lr_rate).minimize(obj_sam, var_list = get_var('discriminator'))

tn = tf.reduce_mean(1.0-p_gen)
tp = tf.reduce_mean(p_sam)

init_op = tf.initialize_all_variables()
sess.run(init_op)
print_variable(-1, 'before training')
start = False
for i in range(n_iter):
    batch_xs, batch_ys = mnist.train.next_batch(n_batch)
    batch_cs = ran(n_batch, n_code)
    true_neg = sess.run(tn, feed_dict={c: batch_cs})
    def train_dis():
        sess.run(dis_step, feed_dict={x_sam: batch_xs, c: batch_cs})
    def train_gen():
        if i<(n_iter/2):
            sess.run(gen_step_early, feed_dict={x_sam: batch_xs, c: batch_cs})
        else:
            sess.run(gen_step, feed_dict={x_sam: batch_xs, c: batch_cs})
    train_dis()
    train_gen()
:   if i%100==0:
        print 'tn->0.5',i, sess.run(tn, feed_dict={x_sam: batch_xs, c: batch_cs})
    #print 'tp->0.5', sess.run(tp, feed_dict={x_sam: batch_xs, c: batch_cs})
    
batch_xs, batch_ys = mnist.train.next_batch(n_batch)
plot_img(sess.run(x_gen, feed_dict={c: ran(3, n_code)}))
#plot_img(sess.run(s_gen, feed_dict={c: ran(100, n_code)}))
#plot_img(sess.run(s_sam, feed_dict={x_sam: batch_xs}))
#print ran(1, n_code)

#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
