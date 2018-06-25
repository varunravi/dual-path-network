# Varun Ravi
# dpn.pyd

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf


def res_net(layer, filter_size, padding='same'):

	layer = tf.layers.conv2d(layer, filter_size, (3,3),activation=tf.nn.relu, padding=padding)
	layer = tf.layers.conv2d(layer, filter_size,(3,3), padding=padding)

	return layer

# def dense_net():



def micro_block(layer, final_filter_size, num_iterations, cardinality=32, activation=None, padding = 'same'):

	for i in range (0, num_iterations):
		initial_filter_size = final_filter_size / (2+2/3)

		layer = tf.layers.conv2d(layer, initial_filter_size, (1,1), activation=activation, padding=padding)
		
		layer_G = []
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

def dpn(X_train, y_train, activation=tf.nn.relu):

	x = tf.placeholder(tf.float32, shape=[X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]], name='x')
	y = tf.placeholder(tf.float32, shape=[y_train.shape[0], y_train.shape[1], y_train.shape[2], y_train.shape[3]], name='y')

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
	layer = tf.contrib.layers.fully_connected(layer, 1000)

	final = tf.layers.conv2d(layer, 2, 1, activation=tf.nn.softmax)

	return final

if __name__ == '__main__':

	X_image = np.zeros((1, 572, 572, 1))
	y_image = np.zeros((1, 572, 572, 1))

	dpn_model = dpn(X_image, y_image)

	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	ipdb.set_trace()

	

