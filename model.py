from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
from operation import *

n_code = 2
n_iter = 550*10000
n_batch = 100
lr_rate = 0.05

n_hidden = [500, 500]

def rep(n_hidden):
    return '_'+"_".join(map(str, n_hidden))

def minimize(obj, wrt):
    return tf.train.AdagradOptimizer(lr_rate).minimize(obj, var_list = wrt)

def op_vae(x_sam, c, op_encoder, op_decoder):
    with tf.variable_scope('vae'):
        with tf.variable_scope('encoder'):
            u, v = op_encoder(x_sam)
        h = u + c * v**0.5
        with tf.variable_scope('decoder'):
            x_rec = op_decoder(h)

    obj_rec = tf.reduce_sum(tf.squared_difference(x_sam, x_rec), reduction_indices = 1)
    obj_reg_avg = tf.reduce_sum(u * u, reduction_indices = 1) * 0.5
    obj_reg_var = tf.reduce_sum(v-tf.log(v)-1, reduction_indices = 1) * 0.5
    obj_vae = obj_rec + obj_reg_avg + obj_reg_var
    print get_var('vae/encoder') 
    step_vae = minimize(obj_vae, get_var('vae/encoder') + get_var('vae/decoder'))
    return (step_vae, obj_vae), (x_rec, h)

def op_gan_boot(x_sam, c, op_encoder, op_decoder):
    # warning: rank of c has to be 2
    with tf.variable_scope('vae'):
        with tf.variable_scope('decoder'):
            x_gen = op_decoder(c)
        with tf.variable_scope('encoder'):
            u_sam, u_gen = op_encoder(x_sam)
            p_sam = tf.nn.softmax(u_sam)[:, 0]
            scope.reuse_variables()
            u_gen, v_gen = op_encoder(x_gen)
            p_gen = tf.nn.softmax(u_gen)[:, 0]

    obj_gen = tf.log(1.0-p_gen) 
    obj_dis = - tf.log(p_sam) - tf.log(1.0-p_gen)
    step_gen = minimize(obj_gen, get_var('vae/decoder'))
    step_dis = minimize(obj_dis, get_var('vae/encoder'))
    return (step_gen, p_gen), (step_dis, p_dis), x_gen
            
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

def model_train_gan((x_sam, c), (step_gen, obj_gen), (step_dis, obj_dis)):
    ud, vd = model_moment()
    saver = tf.train.Saver()
    #min_err = model_error(x_sam, c, err)
    for i in range(n_iter):
        batch_xs, ___ = mnist.train.next_batch(n_batch)
        batch_cs = ran(n_batch, n_code)
        sess.run(step_gen, feed_dict={x_sam: op_stats_apply(batch_xs, ud, vd), c: batch_cs})
        sess.run(step_dis, feed_dict={x_sam: op_stats_apply(batch_xs, ud, vd), c: batch_cs})
       
        one_epoch = mnist.train.images.shape[0]/n_batch
        if i%(one_epoch*10)==0: 
            num_epoch = i/one_epoch

            err_gen = model_error(x_sam, c, obj_gen)
            err_dis = model_error(x_sam, c, obj_dis)
            print num_epoch, err_gen, err_dis
            saver.save(sess, 'model/gan'+rep(n_hidden)+'.ckpt')
            #if cur_err < min_err:
            #    min_err = cur_err
            #    print "saving model"
            #    saver.save(sess, 'model/vae'+rep(n_hidden)+'.ckpt')

def model_train((x_sam, c), (steps, err)):
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
                saver.save(sess, 'model/vae'+rep(n_hidden)+'.ckpt')


def model_plot(x_rec, h, n_dot = 21, n_max = 1, name = ''):
    ud, vd = model_moment()
    img_cols = []
    lin = np.linspace(-n_max, n_max, n_dot)

    for yc in lin:
        pts = np.stack(np.meshgrid(lin, yc), axis = 2).reshape(-1, 2)
        pts = np.concatenate([pts, zero(n_dot, n_code-2)], axis = 1)
        img_cols.append(sess.run(op_stats_unapply(x_rec, ud, vd), feed_dict={h: pts}))
    img_save(img_cols, name)

def model_test(x_sam, c, x_rec, n_view = 5):
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
    img_save([A, B])



def op_encoder(x):
    if n_hidden:
        x = op_vec_full(x, n_hidden, activation = tf.nn.tanh, name = 'hidden')
    u = op_vec_full(x, [n_code], activation = tf.nn.tanh, name = 'mean')
    v = op_vec_full(x, [n_code], activation = tf.nn.sigmoid, name = 'var')
    return u, v

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

x = tf.placeholder(tf.float32, [None,  784])
c = tf.placeholder(tf.float32, [None, n_code])

(step_vae, obj_vae), (x_rec, h) = op_vae(x, c, op_encoder, op_decoder)
#(step_vae, obj_vae), (step_vae, obj_vae), x_rec = op_vae(x, c, op_encoder, op_decoder)
#step_gen, step_dis = op_gan(x, c, op_generator, op_discriminator)

'''
try:
    model_load('model/vae'+rep(n_hidden)+'.ckpt')
except:
    model_init()
model_train((x, c), (step_vae, obj_vae))
#model_save('model/vae'+rep(n_hidden)+'.ckpt')

'''
model_load('model/vae'+rep(n_hidden)+'.ckpt')
model_test(x, c , x_rec)
model_plot(x_rec, h)
model_plot(x_rec, h, name = 'vae')
