# Varun Ravi
# dpn.py

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf



def micro_block(layer, final_filter_size, num_iterations, activation=tf.nn.relu):

	layer_dense_path = tf.placeholder(tf.float32)
	layer_res_path = tf.placeholder(tf.float32)

	layer_dense_path = layer
	layer_res_path = layer
	
	
	for i in range(0, num_iterations):
		layer = tf.layers.conv2d(layer, final_filter_size/(2+2/3), (1,1), activation=activation)
		layer = tf.layers.conv2d(layer, final_filter_size/(2+2/3), (3,3), activation=activation)
		layer = tf.layers.conv2d(layer, final_filter_size, (1,1), activation=activation)

	layer_dense_path = tf.image.resize_nearest_neighbor(layer_dense_path, (layer.get_shape().as_list()[1],layer.get_shape().as_list()[2]))
	layer_res_path = tf.image.resize_nearest_neighbor(layer_res_path, (layer.get_shape().as_list()[1],layer.get_shape().as_list()[2]))
	

	return tf.concat([layer, layer_dense_path, layer_res_path], axis=-1)


def dpn(X_train, y_train, activation=tf.nn.relu):

	x = tf.placeholder(tf.float32, shape=[X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]], name='x')
	y = tf.placeholder(tf.float32, shape=[y_train.shape[0], y_train.shape[1], y_train.shape[2], y_train.shape[3]], name='y')

	# conv1
	conv1 = tf.layers.conv2d(x, 64, (7,7), activation=activation)
	
	# conv2
	conv2 = tf.layers.conv2d(conv1, 64, (3,3), activation=activation)
	conv2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(3,3), strides=2)
	
	conv2 = micro_block(conv2, 256, 3)
	conv3 = micro_block(conv2, 512, 4)
	conv4 = micro_block(conv3, 1024, 20)
	conv5 = micro_block(conv4, 2048, 3)

	final = tf.layers.conv2d(conv5, 2, 1, activation=tf.nn.softmax)

	ipdb.set_trace()


if __name__ == '__main__':

	X_image = np.zeros((1, 572, 572, 1))
	y_image = np.zeros((1, 572, 572, 1))

	dpn(X_image, y_image)