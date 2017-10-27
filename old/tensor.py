import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

"""
Input > weight > hidden layer 1 (activation function) > weights > hidden l 2.
repeat until output layer.

compare output to intended output > cost function (cross entropy)

optimization function (optimizer) > minimise the cost (AdamOptimiser, ...SGC, AdaGrad)

feed forward + backprop = epoch
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
...
'''
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# input feature size = 28x28 pixels = 784
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# (input_data * weights) + biases
	hidden_1_layer =  {
		'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
	    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
	}
	hidden_2_layer =  {
		'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    	'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
	hidden_3_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    	'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }
	output_layer =  {
		'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    	'biases': tf.Variable(tf.random_normal([n_classes]))
    }

	# (input_data * weights) + biases
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	return output

def train_neural_network(x,y):
	prediction = neural_network_model(x)
	# compare the cost with the prediction.
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # like sochastic gradient descent
	# the learning rate is like 0.01
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles of feed forward + backprop
	hm_epochs = 10

	with tf.Session() as sess:
		# initialises all variables;
		# session is now running
		sess.run(tf.global_variables_initializer())
		# training the network.
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x,y:epoch_y})
				epoch_loss +=c 
			print('Epoch', epoch, 'completed out of', hm_epochs, "loss:", epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x,y)