def save(checkpoint_file=hello.chk):
    with tf.Session() as session:
        x = tf.Variable([42.0, 42.1, 42.3], name=x)
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name=y)
        not_saved = tf.Variable([-1, -2], name=not_saved)
        session.run(tf.initialize_all_variables())

        print(session.run(tf.all_variables()))
        saver = tf.train.Saver([x, y])
        saver.save(session, checkpoint_file)

def restore(checkpoint_file=hello.chk):
    x = tf.Variable(-1.0, validate_shape=False, name=x)
    y = tf.Variable(-1.0, validate_shape=False, name=y)
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file)
        print(session.run(tf.all_variables()))

def reset():
    tf.reset_default_graph()

