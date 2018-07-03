# Varun Ravi
# dpn.py

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


BATCH_SIZE = 8
# img_height = 32
# img_width = 32
# num_channels = 3


def res_net(layer, filter_size, padding='same'):

	layer = tf.layers.conv2d(layer, filter_size, (3,3),activation=tf.nn.relu, padding=padding)
	layer = tf.layers.conv2d(layer, filter_size,(3,3), padding=padding)

	return layer

def micro_block(layer, final_filter_size, num_iterations, cardinality=32, activation=None, padding='same'):

	for i in range (0, num_iterations):
		layer_G = []

		initial_filter_size = final_filter_size / (2+2/3)
		layer = tf.layers.conv2d(layer, initial_filter_size, (1,1), activation=activation, padding=padding)
		for i in range(0, 32):
			layer_G.append(tf.layers.conv2d(layer, initial_filter_size, (3,3), activation=activation, padding=padding))

		layer = tf.concat(values=layer_G, axis=3, name='concat')
		layer = tf.layers.conv2d(layer, 256, (1,1), activation=activation, padding=padding)

	return layer

def macro_block(dense_net, layer, filter_size, micro_iter):
	
	layer = micro_block(layer, filter_size, micro_iter)
	res_layer = res_net(layer, filter_size)
	dense_layer = tf.concat([dense_net, layer], axis=3)
	layer = tf.concat([dense_layer, res_layer], axis=3)

	return [layer,dense_layer]

def one_hot(labels, shape):
	one_hot= np.zeros(shape)
	pos=0

	for i in labels:
		one_hot[pos][i] = 1
		pos+=1

	return one_hot


if __name__ == '__main__':

	#model 
	x_shape = [None, 32, 32, 3]
	y_shape = [None, 10]
	x = tf.placeholder(tf.float32, shape=x_shape, name='x')
	y = tf.placeholder(tf.float32, shape=y_shape, name='y')

	conv1 = tf.layers.conv2d(x, 64, (2,2), (2,2), activation=tf.nn.relu, padding='same')
	conv2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(3,3), strides=2, padding='same')
	conv2 = micro_block(conv2, 256, 3)
	res_conv2 = res_net(conv2, 256)
	dense_net = conv2
	conv2 = tf.concat([res_conv2, conv2], axis=3)
	layer, dense_net = macro_block(dense_net, conv2, 512, 4)
	layer, dense_net = macro_block(dense_net, layer, 1024, 20)
	layer, _ = macro_block(dense_net, layer, 2048, 3)
	layer = tf.layers.average_pooling2d(layer, (2,2), 2)
	ipdb.set_trace()
	layer = tf.contrib.layers.flatten(layer)
	layer = tf.layers.dense(inputs=layer, units=1000, activation=tf.nn.relu)
	final = tf.layers.dense(inputs=layer, units=10, activation=tf.nn.softmax)
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final, labels=y))
	optimize = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

	#optimize = tf.train.AdamOptimizer().minimize(loss)

	#ipdb.set_trace()
	# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	# train_data = mnist.train.images # Returns np.array
	# train_labels = np.asarray(mnist.train.labels)
	# eval_data = mnist.test.images # Returns np.array
	# eval_labels = np.asarray(mnist.test.labels)
	# train_data = train_data.reshape(55000,28,28,1)
	# eval_data = eval_data.reshape(10000,28,28,1)

	(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.cifar10.load_data()
	train_labels = one_hot(train_labels, (50000,10))
	eval_labels = one_hot(eval_labels, (10000,10))

	init = tf.global_variables_initializer()	
	saver = tf.train.Saver()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement=True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.7

	with tf.Session(config=config) as session:
		session.run(init)
		for epoch in range(0, 1):
			total_steps = int(len(train_data)/BATCH_SIZE)
			for step in range(10):
				start = step*BATCH_SIZE
				end = start+BATCH_SIZE
				X_batch = train_data[start:end]
				y_batch = train_labels[start:end]

				_, training_loss = session.run([optimize, loss], feed_dict={x: X_batch, y: y_batch})
				
				print ("Steps %d/%d, Training Loss=%f" % (step+1, total_steps, training_loss))

			total_steps = int (len(eval_data)/BATCH_SIZE)
			total_loss = 0
			for step in range(20):
				start = step*BATCH_SIZE
				end = start + BATCH_SIZE
				X_batch = eval_data[start:end]
				y_batch = eval_labels[start:end]
				valid_loss = session.run(loss, feed_dict={x: X_batch, y: y_batch})
				total_loss += valid_loss
				print ("Steps %d/%d, Validation Loss=%f" % (step+1, total_steps, valid_loss))
			valid_loss = total_loss / total_steps
			print ("Epoch %d, Training Loss = %d, Validation Loss = %d" % (epoch, training_loss, valid_loss))
		saver.save(session, './dpn_results-tf')

		X_batch = eval_data[0:10]
		y_batch = eval_labels[0:10]

		y_pred = session.run(final, feed_dict={x: X_batch, y: y_batch})
		ipdb.set_trace()
		print(y_pred)
		
		

