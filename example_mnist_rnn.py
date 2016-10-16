import tensorflow as tf

num_neurons = 200
num_layers = 3
dropout = tf.placeholder(tf.float32)

cell = tf.nn.rnn_cell.GRUCell(num_neurons)  # Or LSTMCell(num_neurons)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

num_neurons = 300
cell2 = tf.nn.rnn_cell.GRUCell(num_neurons)  # Or LSTMCell(num_neurons)
#cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=dropout)
#cell2 = tf.nn.rnn_cell.MultiRNNCell([cell2] * num_layers)
max_length = 100

# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, max_length, 28])
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
output2, state2 = tf.nn.dynamic_rnn(cell2, output, dtype=tf.float32)





print 'data', data.get_shape()
print 'state', state.get_shape()
print 'output', output.get_shape()
print 'state2', state2.get_shape()
print 'output2', output2.get_shape()
