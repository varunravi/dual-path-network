# Varun Ravi
# dpn.py

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
BATCH_SIZE = 32

def res_net(layer, filter_size, padding='same'):

	layer = tf.layers.conv2d(layer, filter_size, (3,3),activation=tf.nn.relu, padding=padding)
	layer = tf.layers.conv2d(layer, filter_size,(3,3), padding=padding)

	return layer

# def dense_net():



def micro_block(layer, final_filter_size, num_iterations, cardinality=32, activation=None, padding='same'):

	for i in range (0, num_iterations):
		initial_filter_size = final_filter_size / (2+2/3)

		layer = tf.layers.conv2d(layer, initial_filter_size, (1,1), activation=activation, padding=padding)
		
		layer_G = []
		for i in range(0, 32):
			layer_G.append(tf.layers.conv2d(layer, initial_filter_size, (3,3), activation=activation, padding=padding))
		#ipdb.set_trace()
		layer = tf.concat(values=layer_G, axis=3, name='concat')
		layer = tf.layers.conv2d(layer, 256, (1,1), activation=activation, padding=padding)



	return layer

def macro_block(dense_net, layer, filter_size, micro_iter):
	
	layer = micro_block(layer, filter_size, micro_iter)
	res_layer = res_net(layer, filter_size)
	dense_layer = tf.concat([dense_net, layer], axis=3)
	layer = tf.concat([dense_layer, res_layer], axis=3)

	return [layer,dense_layer]

def dpn2(x=None, activation=tf.nn.relu):

	# mnist = tf.contrib.learn.datasets.load_dataset("mnist", one_hot=True)
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels)


	
	x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name='x')
	y = tf.placeholder(tf.float32, [None, 10], 'y')

	#with tf.Graph().as_default() as graph:

	# x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name='x')
	# y = tf.placeholder(tf.float32, [None, 1], 'y')
	
	conv1 = tf.layers.conv2d(x, 64, (7,7), (2,2), activation=activation, padding='same')
	layer = tf.contrib.layers.flatten(conv1)
	layer = tf.contrib.layers.fully_connected(layer, 1000)		# conv1
	final = tf.contrib.layers.fully_connected(layer, 10)

	#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.cast(tf.argmax(final, output_type=tf.int32), dtype=tf.float32), labels=tf.cast(y, dtype=tf.float32)))
		#loss_op = tf.nn.softmax(dpn_model)

	return x, y, final

def dpn(x=None, activation=tf.nn.relu):
	
	img_height = 28
	img_width = 28
	num_channels = 1
	y_shape = 55000

	x = tf.placeholder(tf.float32, shape=[None, img_height, img_width, num_channels], name='x')
	y = tf.placeholder(tf.float32, [None, 10], 'y')

	# conv1
	conv1 = tf.layers.conv2d(x, 64, (7,7), (2,2), activation=activation, padding='same')
	conv2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(3,3), strides=2, padding='same')

	# conv2
	conv2 = micro_block(conv2, 256, 3)
	res_conv2 = res_net(conv2, 256)
	dense_net = conv2
	conv2 = tf.concat([res_conv2, conv2], axis=3)

	layer, dense_net = macro_block(dense_net, conv2, 512, 4)
	layer, dense_net = macro_block(dense_net, layer, 1024, 20)
	layer, _ = macro_block(dense_net, layer, 2048, 3)

	layer = tf.layers.average_pooling2d(layer, (2,2), 2)
	
	layer = tf.contrib.layers.flatten(layer)
	final = tf.contrib.layers.fully_connected(layer, 1000)
	final = tf.contrib.layers.fully_connected(final, 10)

	return x, y, final

if __name__ == '__main__':

	img_height = 28
	img_width = 28
	num_channels = 1
	y_shape = 55000

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels)

	train_data = train_data.reshape(55000,img_height,img_width,num_channels)
	eval_data = eval_data.reshape(10000,img_height,img_width,num_channels)	
	
	x, y, final = dpn()

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final, labels=y))
	train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

	init = tf.global_variables_initializer()	

	with tf.Session() as session:
		session.run(init)
		for epoch in range(0, 2):
			total_steps = int(len(train_data)/BATCH_SIZE)
			for step in range(total_steps):
				start = step*BATCH_SIZE
				end = start + BATCH_SIZE
				X_batch = train_data[start:end]
				y_batch = train_labels[start:end]

				_, training_loss = session.run([train, loss], feed_dict={x: X_batch, y: y_batch})
				
				print (step+1, total_steps, training_loss)

			total_steps = int (len(eval_data)/BATCH_SIZE)
			total_loss = 0
			for step in range(total_steps):
				start = step*BATCH_SIZE
				end = start + BATCH_SIZE
				X_batch = eval_data[start:end]
				y_batch = eval_labels[start:end]
				valid_loss = session.run(loss, feed_dict={x: X_batch, y: y_batch})
				total_loss += valid_loss	
			valid_loss = total_loss / total_steps
			print ("Epoch %d, Training Loss = %d, Validation Loss = %d" % (epoch, training_loss, valid_loss))
		#saver.save(session, './dpn_results-tf')



	#ipdb.set_trace()

	

