from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
from operation import *
import matplotlib.pylab as plt
n_code = 2
n_iter = 550*10000
n_batch = 100
lr_rate = 0.001
l2_reg = 0.001
n_hidden = [500, 500]

exp_name = 'vae_2dcov'
def rep(n_hidden):
    return '_'+"_".join(map(str, n_hidden))

def minimize(obj, wrt):
    #return tf.train.AdagradOptimizer(lr_rate).minimize(obj, var_list = wrt)
    return tf.train.AdamOptimizer(lr_rate).minimize(obj, var_list = wrt)

def op_encoder(x):
    if n_hidden:
        x = op_vec_full(x, n_hidden, activation = tf.nn.tanh, name = 'hidden')
    u = op_vec_full(x, [n_code], activation = tf.nn.tanh, name = 'mean')
    v = op_vec_full(x, [n_code], activation = tf.nn.sigmoid, name = 'var')
    return u, v

def op_encoder_full(x):
    if n_hidden:
        x = op_vec_full(x, n_hidden, activation = tf.nn.tanh, name = 'hidden')
    u = op_vec_full(x, [n_code], activation = tf.nn.tanh, name = 'mean')
    #v = op_vec_full(x, [n_code*(n_code+1)/2], activation = tf.nn.sigmoid, name = 'var')
    v = op_vec_full(x, [n_code*(n_code+1)/2], activation = tf.nn.relu, name = 'var')
    C = tf.get_variable("cov_mat", [n_code, n_code])
    cov_list = []
    dia_list = []
    for i in range(n_code):
        for j in range(n_code):
            if i>=j:
                cov_list.append(v[:,i*(i+1)/2+j])
            else:
                cov_list.append(v[:,j*(j+1)/2+i])
            if i==j:
                dia_list.append(v[:,i*(i+1)/2+j])
    C = tf.reshape(tf.transpose(tf.pack(cov_list), perm = [1, 0]), [-1, n_code, n_code])
    dia = tf.reshape(tf.transpose(tf.pack(dia_list), perm = [1, 0]), [-1, n_code])
    return u, C, dia, v

def op_decoder(c):
    x = op_vec_full(c, n_hidden[::-1] + [784], activation = tf.nn.tanh, name = 'reconstruction')
    return x

def op_generator(c):
    # input sample from random noise
    # output counterfiet example
    c = op_vec_full(c, [100], activation = tf.nn.relu, name = 'relu')
    x = op_vec_full(c, [784], activation = tf.nn.sigmoid, name = 'sig')
    return x

def op_discriminator(x):
    # input sample to be judged
    # output probability p for passing the discriminator test
    x = op_vec_full(x, [100], activation = tf.nn.relu, name = 'relu')
    p = op_vec_full(x, [1], activation = tf.nn.sigmoid, name = 'sig')
    return p

def l2_loss(scope):
    return 0
    #return l2_reg * tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    #weights = get_var(scope)
    losses = [tf.nn.l2_loss(w) for w in weights]
    total_loss = tf.add_n(losses) 
    return total_loss

def op_vae(x_sam, c, op_encoder, op_decoder):
    with tf.variable_scope('vae'):
        with tf.variable_scope('encoder'):
            u, v = op_encoder(x_sam)
        h = u + c * v**0.5
        with tf.variable_scope('decoder'):
            x_rec = op_decoder(h)


def batch_map(C, op):
    #what is going on
    #input: tensors of shape [?, n_col, n_row]
    #ouput: op applied to every matrix then concatenated, final shape [?, op(C)]
    print C.get_shape()
    mat_list = tf.unpack(C, axis = 0)
    x = tf.trace(tf.constant([[[1, 2, 3],[4, 5, 6],[7, 8, 9]], [[1, 2, 3],[4, 5, 6],[7, 8, 9]]], dtype=tf.int32))
    print sess.run(x)
    temp = tf.map_fn(tf.trace, mat_list)
    temp2 = tf.pack(temp)

    return tf.transpose(temp2, perm = [0,1,2])

def three_x_two(A, B):
    #Cijl = np.matmul(Aijk,Bkl)
    #tf.reshape(tf.matmul(tf.reshape(Aijk,[i*j,k]),Bkl),[i,j,l])
    print A.get_shape()
    print B.get_shape()
    i = tf.shape(A)[0]
    j = tf.shape(A)[1]
    k = tf.shape(A)[2]
    l = tf.shape(B)[1]
    return tf.reshape(tf.matmul(tf.reshape(A,[i*j,l]),B) ,  [i, j, l])


def det2d(v):
    #input: [a  b
    #        b  c ]
    #output: ac-b^2
    return v[:,0]*v[:,2]-v[:,1]*v[:,1]

def op_vae_full(x_sam, c, op_encoder, op_decoder):
    with tf.variable_scope('vae'):
        with tf.variable_scope('encoder'):
            u, C, dia, flat = op_encoder(x_sam)

        #temp = tf.tile(c, [tf.shape(C)[0], 1])
        #temp2 = tf.reshape(temp, [tf.shape(C)[0],-1,1]) 
        #tf.reshape(tf.batch_matmul(C **0.5,  temp2), [-1])
        
        temp = tf.reshape(c, [-1, n_code, 1]) 
        v = tf.reshape(tf.batch_matmul(C **0.5,  temp), [-1, n_code])
        
        #v = tf.reshape(three_x_two(C**0.5, tf.reshape(c, [-1, 1])), [-1, tf.shape(c)[0]])

        h = u + v

        with tf.variable_scope('decoder'):
            x_rec = op_decoder(h)

    obj_rec = tf.reduce_sum(tf.squared_difference(x_sam, x_rec), reduction_indices = 1)
    obj_reg_avg = tf.reduce_sum(u * u, reduction_indices = 1) * 0.5
    #obj_reg_var = tf.reduce_sum(v-tf.log(v)-1, reduction_indices = 1) 
    obj_reg_var = tf.reduce_sum(dia, reduction_indices=1) - tf.log(det2d(flat)+0.00001)  # tf.log(tf.matrix_determinant(C))
    obj_vae = obj_rec + obj_reg_avg + 0.5*obj_reg_var 
    step_vae = minimize(obj_vae+ l2_loss('vae'), get_var('vae/encoder') + get_var('vae/decoder'))
    return step_vae, obj_vae, x_rec, h, u, C

def op_ganvae(x_sam, c, op_encoder, op_decoder):
    # warning: rank of c has to be 2
    with tf.variable_scope('vae'):
        with tf.variable_scope('decoder'):
            x_gen = op_decoder(c)
        with tf.variable_scope('encoder') as scope:
            u_gen, v_gen = op_encoder(x_gen)
            p_gen = tf.nn.softmax(u_gen)[:, 0]
            
            scope.reuse_variables()
            u_sam, u_gen = op_encoder(x_sam)
            p_sam = tf.nn.softmax(u_sam)[:, 0]

    obj_gen = tf.log(1.0-p_gen) 
    #obj_gen = -tf.log(p_gen)
    obj_dis = - tf.log(p_sam) - tf.log(1.0-p_gen) 
    step_gen = minimize(obj_gen+ l2_loss('vae/decoder'), get_var('vae/decoder'))
    step_dis = minimize(obj_dis+ l2_loss('vae/encoder'), get_var('vae/encoder'))
    return step_gen, step_dis, p_gen, p_sam, x_gen
            
def op_gan(x_sam, c, op_generator, op_discriminator):
    with tf.variable_scope('gan'):
        with tf.variable_scope('generator'):
            x_gen = op_generator(c)
        with tf.variable_scope('discriminator') as scope:
            p_sam = op_discriminator(x_sam)
            scope.reuse_variables()
            p_gen = op_discriminator(x_gen)

    obj_gen = tf.log(1.0-p_gen) 
    obj_sam = - tf.log(p_sam) - tf.log(1.0-p_gen)
    step_gen = minimize(obj_gen, get_var('gan/generator'))
    step_dis = minimize(obj_dis, get_var('gan/discriminator'))
    return (step_gen, obj_gen), (step_dis, obj_dis), x_gen

def model_moment():
    ___ = tf.placeholder(tf.float32, [None,  784])
    ud, vd = sess.run(op_stats_get(___), feed_dict={___: mnist.train.images})
    return ud, vd

def model_save(model_name):
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_name)

def model_load(model_name): 
    saver = tf.train.Saver()
    saver.restore(sess, model_name)

def model_init():
    sess.run(tf.initialize_all_variables())

def model_error(x_sam, c, err):
    ud, vd = model_moment()
    train_err = sess.run(
        tf.reduce_mean(err), 
        feed_dict={
            x_sam: op_stats_apply(mnist.train.images, ud, vd), 
            c: zero(mnist.train.images.shape[0], n_code)
        }
    )
    return train_err

def model_train_gan(x_sam, c, step_gen, step_dis, p_gen, p_sam):
    ud, vd = model_moment()
    saver = tf.train.Saver()
    #min_err = model_error(x_sam, c, err)
    for i in range(n_iter):
        batch_xs, ___ = mnist.train.next_batch(n_batch)
        batch_cs = ran(n_batch, n_code)
        sess.run(step_dis, feed_dict={x_sam: op_stats_apply(batch_xs, ud, vd), c: batch_cs})
        sess.run(step_gen, feed_dict={x_sam: op_stats_apply(batch_xs, ud, vd), c: batch_cs})
       
        one_epoch = mnist.train.images.shape[0]/n_batch
        if i%(one_epoch*10)==0: 
            num_epoch = i/one_epoch

            e_gen = model_error(x_sam, c, p_gen)
            e_sam = model_error(x_sam, c, p_sam)
            print num_epoch, e_gen, e_sam
            with open("model/gan"+rep(n_hidden)+'.loss', "a") as myfile:
                myfile.write('{} {}\n'.format(e_gen, e_sam))
            saver.save(sess, 'model/gan'+rep(n_hidden)+'.ckpt')
            #if cur_err < min_err:
            #    min_err = cur_err
            #    print "saving model"
            #    saver.save(sess, 'model/vae'+rep(n_hidden)+'.ckpt')

def model_train(x_sam, c, steps, err):
    ud, vd = model_moment()
    saver = tf.train.Saver()
    min_err = model_error(x_sam, c, err)
    for i in range(n_iter):
        batch_xs, ___ = mnist.train.next_batch(n_batch)
        batch_cs = ran(n_batch, n_code)
        sess.run(steps, feed_dict={x_sam: op_stats_apply(batch_xs, ud, vd), c: batch_cs})
       
        one_epoch = mnist.train.images.shape[0]/n_batch
        if i%(one_epoch*10)==0: 
            num_epoch = i/one_epoch

            cur_err = model_error(x_sam, c, err)
            print num_epoch, cur_err
            if cur_err < min_err:
                min_err = cur_err
                print "saving model"
                saver.save(sess, 'model/'+exp_name+''+rep(n_hidden)+'.ckpt')


def img_circle(u, v, text =True,  **args):
    from matplotlib.patches import Ellipse
    e = Ellipse((u[0], u[1]), width = v[0]*2, height = v[1]*2, fill=False, linewidth=2, **args)
    ax = plt.gca()
    ax.add_artist(e)
    if text:
        ax.text(u[0], u[1]+v[1],  str(v[1])+' std', 
            size = 20, 
            verticalalignment='bottom', 
            horizontalalignment='center', **args)

def filt_small_var(x):
    u, v = x
    return u+v>0.5

def img_elli(x_sam, x_rec, u, v, n_view = 5, view = 'train', filt_crit = lambda ___: True):
    ud, vd = model_moment()
    if view=='train':
        batch_xs, ___ = mnist.train.next_batch(n_view)
    else:
        batch_xs, ___ = mnist.test.next_batch(n_view)
        
    X = batch_xs
    X_, U, V = sess.run([op_stats_unapply(x_rec, ud, vd), u, v], 
            feed_dict={ x_sam: op_stats_apply(batch_xs, ud, vd), 
                        c: zero(n_view, n_code)})
    #print U, V
    for ui, vi in filter(filt_crit, zip(U, V)):
        img_circle(ui, map(np.sqrt, vi), text=False, color = 'b', alpha= 1.0/(1.0+np.log(n_view)/2))
    
def img_recon(x_sam, x_rec, n_view = 5, view = 'train'):
    ud, vd = model_moment()
    if view=='train':
        batch_xs, ___ = mnist.train.next_batch(n_view)
    else:
        batch_xs, ___ = mnist.test.next_batch(n_view)
    X = batch_xs
    X_ = sess.run(op_stats_unapply(x_rec, ud, vd), 
            feed_dict={ x_sam: op_stats_apply(batch_xs, ud, vd), 
                        c: zero(n_view, n_code)})
    matrix = []
    for i,x in enumerate([X, X_]):
        matrix.append(np.reshape(x, [-1, 28]))
    mat = np.concatenate(matrix, axis = 1)
    plt.imshow(1-mat, interpolation='none', cmap=plt.cm.gray)

def img_code(xs, h, n_max=2, n_dot=21, quart=False):
    ud, vd = model_moment()
    if quart: nn_max = 0
    else: nn_max = -n_max
    lin = np.linspace(nn_max, n_max, n_dot)

    img_cols = []
    for yc in lin:
        pts = np.stack(np.meshgrid(lin, yc), axis = 2).reshape(-1, 2)
        pts = np.concatenate([pts, zero(n_dot, n_code-2)], axis = 1)
        img_cols.append(sess.run(op_stats_unapply(x_rec, ud, vd), feed_dict={h: pts}))

    matrix = []
    for i,x in enumerate(img_cols):
        matrix.append(np.reshape(x, [-1, 28]))
    mat = np.concatenate(matrix, axis = 1)
    
    plt.imshow(1-mat, interpolation='none', cmap=plt.cm.gray, extent = [nn_max, n_max , nn_max, n_max])
    plt.xticks(np.linspace(nn_max, n_max, 5))
    plt.yticks(np.linspace(nn_max, n_max, 5))

    for i in range(int(n_max)):
        img_circle([0,0], [i+1,i+1], color = 'r')



def model_plot(x_rec, h, n_dot = 21, n_max = 1, quart = False, name = ''):
    ud, vd = model_moment()
    img_cols = []
    if quart:
        lin = np.linspace(0, n_max, n_dot)
    else:
        lin = np.linspace(-n_max, n_max, n_dot)

    for yc in lin:
        pts = np.stack(np.meshgrid(lin, yc), axis = 2).reshape(-1, 2)
        pts = np.concatenate([pts, zero(n_dot, n_code-2)], axis = 1)
        img_cols.append(sess.run(op_stats_unapply(x_rec, ud, vd), feed_dict={h: pts}))
    img_save(img_cols, name = name)

def model_test(x_sam, c, x_rec, n_view = 5, name = ''):
    ___ = tf.placeholder(tf.float32, [None,  784])
    ud, vd = sess.run(op_stats_get(___), feed_dict={___: mnist.train.images})
    batch_xs, ___ = mnist.train.next_batch(n_view)
    A = batch_xs
    B = sess.run(op_stats_unapply(x_rec, ud, vd), 
            feed_dict={
                x_sam: op_stats_apply(batch_xs, ud, vd), 
                c: zero(n_view, n_code)
            }
        )
    img_save([A, B], name)


x = tf.placeholder(tf.float32, [None,  784])
c = tf.placeholder(tf.float32, [None, n_code])



GAN, VAE = False, True
#GAN, VAE = True, False
#TRAIN, IMG = False, True
TRAIN, IMG = True, False

if GAN:
    step_gen, step_dis, p_gen, p_sam, x_gen = op_ganvae(x, c, op_encoder, op_decoder)
    print [w.name for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vae/encoder')]
    x_rec, h = x_gen, c

    try: model_load('model/gan'+rep(n_hidden)+'.ckpt')
    except: model_init()

    if TRAIN: model_train_gan(x, c, step_gen, step_dis, p_gen, p_sam)
elif VAE:
    #step_vae, obj_vae, x_rec, h, u, v = op_vae(x, c, op_encoder, op_decoder)
    step_vae, obj_vae, x_rec, h, u, v = op_vae_full(x, c, op_encoder_full, op_decoder)

    try: model_load('model/'+exp_name+''+rep(n_hidden)+'.ckpt')
    except: model_init()

    if TRAIN: model_train(x, c, step_vae, obj_vae)

if IMG:

    def image_0():
        plt.figure(num=None, figsize=(10, 10), dpi=80)
        img_code(x_rec, h, n_max = 3, n_dot = 41)
        plt.savefig('code_3std.png')
        plt.show()

    def image_1():
        plt.figure(num=None, figsize=(10, 10), dpi=80)
        img_code(x_rec, h, n_max = 1, n_dot = 15)
        plt.savefig('code_1std.png')
        plt.show()
    
    def image_2():
        plt.figure(num=None, figsize=(10, 10), dpi=80)
        img_code(x_rec, h, n_max = 1, n_dot = 15, quart = False)
        img_elli(x, x_rec, u, v, n_view = 20)
        plt.savefig('train_elli_small.png')
        plt.show()

    def image_3():
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        img_code(x_rec, h, n_max = 3, n_dot = 41, quart = False)
        img_elli(x, x_rec, u, v, n_view = 1000, view= 'train')
        plt.savefig('train_elli.png')
        plt.show()

    def image_4():
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        img_code(x_rec, h, n_max = 3, n_dot = 41, quart = False)
        img_elli(x, x_rec, u, v, n_view = 1000, view='test')
        plt.savefig('test_elli.png')
        plt.show()

    def image_5():
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        img_recon(x, x_rec, n_view = 10, view = 'train')
        plt.savefig('train_recon.png')
        plt.show()

    def image_6():
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        img_recon(x, x_rec, n_view = 10, view = 'test')
        plt.savefig('test_recon.png')
        plt.show()
    
    def image_7():
        plt.figure(num=None, figsize=(20, 20), dpi=80)
        img_code(x_rec, h, n_max = 3, n_dot = 41, quart = False)
        img_elli(x, x_rec, u, v, n_view = 10000, view='train', filt_crit = lambda x: x[1][0]>0.5 or x[1][1]>0.5)
        plt.savefig('train_extreme.png')
        plt.show()
   
    image_0() 
    image_1() 
    image_2() 
    image_3() 
    image_4() 
    image_5() 
    image_6() 
    image_7() 
