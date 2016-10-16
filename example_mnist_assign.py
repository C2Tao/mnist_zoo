import tensorflow as tf
sess = tf.Session()
v = tf.Variable(0)
new_v = v.assign(10)
output = new_v + 5  # `new_v` is evaluated after the assignment.

#sess.run(v.initializer)
sess.run(tf.initialize_all_variables())
result = sess.run([output])
print result  # ==> 15
