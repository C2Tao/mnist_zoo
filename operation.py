import tensorflow as tf
import numpy as np
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

def get_var(scope):
    return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)

def assign_var(name, value):
    assign_op = x.assign(value)
    sess.run(assign_op)

def get_scope(scope):
    #return tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope)
    #return [v for v in tf.all_variables() if v.name == vname][0]
    #return  tf.get_default_graph().get_tensor_by_name(vname)
    
    #if not scope: return tf.all_variables()
    #return [v for v in tf.all_variables() if v.name.startswith(scope)]
    pass

def print_variable(i, comment = ''):
    variables = tf.get_collection(tf.GraphKeys.VARIABLES)
    print comment
    print map(lambda g: g.name, variables)
    print map(lambda g: g.get_shape(), variables)
    if i>0:
        print variables[i].name
        print sess.run(variables[i])

def op_reshape(x, shape):
    y = tf.reshape(x, shape)
    return y

def op_img_pool(x):
    # x: [?, x_width, x_height, x_channel]
    # y: [?, x_width/2, x_height/2, x_channel]
    y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
    return y

def op_img_conv(x, n_width, n_height, n_output):
    # x: [?, x_width, x_height, n_input]
    # y: [?, x_width, x_height, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope('conv'):
        w = var('w', ran(n_width, n_height, n_input, n_output)*0.001)
        b = var('b', zero(n_output))
        y = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b
    return y

def op_img_full(x, n_hidden, name = 'cnn', n_width = 5, n_height = 5):
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

def op_img_deconv(x, n_width, n_height, n_output):
    n_input = x.get_shape()[-1]
    with tf.name_scope('deconv') as scope:
        y = tf.nn.conv2d_transpose(x, w, [1, 1, 1, 1], [1, 2, 2, 1], padding='SAME')
    return y

def op_vec_full(x, n_hidden, name = 'full', activation = tf.nn.relu):
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


def op_vec_logr(x, n_output):
    # x: [?, n_input]
    # y: [?, n_output]
    n_input = x.get_shape()[-1]
    with tf.variable_scope('logr'):
        w = var('w', ran(n_input, n_output))
        b = var('b', ran(n_output))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
    return y

def op_seq_lstm(x, n_output):
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

def op_xent(y, y_):
    # y: [?, n_feat]
    # y_: [?, n_feat]
    # e: scalar
    e = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    return e


def op_net_conv(x, y_, name):
    with tf.variable_scope(name):
        x = op_reshape(x, [-1, 28, 28, 1])
        x = op_img_conv(x, 5, 5, 32)
        x = op_img_pool(x)
        x = op_reshape(x, [-1, 14 * 14 * 32])
        y = op_vec_logr(x, 10)
        e = op_xent(y, y_)
    return y, e

def op_net_logr(x,  y_, name):
    with tf.variable_scope(name):
        y = op_vec_logr(x, 10)
        e = op_xent(y, y_)
    return y, e


def op_net_lstm(x, y_, name):
    with tf.variable_scope(name):
        x = op_reshape(x, [-1, 28, 28]) # 28 frames of 28 dim feat
        x = op_seq_lstm(x, 5) # 28 frames of 5 dim feat
        x = op_reshape(x, [-1, 5 * 28])
        y = op_vec_logr(x, 10)
        e = op_xent(y, y_)
    return y, e

def op_kldiv(u, s):
    return 0.5 * tf.reduce_sum(u*u, reducde_indices = 1)

def op_stats_get(x):
    return tf.nn.moments(x, axes = [0])

def op_stats_apply(x, u, v):
    #return tf.nn.batch_normalization(x, u, v, offset = None, scale = None, variance_epsilon = 0.001)
    return (x - u)/ (v**0.5+0.0001)

def op_stats_unapply(x, u, v):
    return x * (v**0.5 + 0.0001) + u

def img_save(xs, name = None):
    if name:
        print "save image to: "+name+'.png'
        import matplotlib
        matplotlib.use('Agg')
    else:
        print "show image only"
    import matplotlib.pylab as plt
    matrix = []
    for i,x in enumerate(xs):
        matrix.append(np.reshape(x, [-1, 28]))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    #print matrix[0].shape
    #print matrix[1].shape
    mat = np.concatenate(matrix, axis = 1)
    print mat.shape
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.ocean)
    if name:
        plt.savefig(name+'.png')
    else:
        plt.colorbar()
        plt.show()

def img_plot(x):
    matrix = np.reshape(x, [-1, 28])
    import matplotlib.pylab as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()

