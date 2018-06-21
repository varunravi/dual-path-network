# Varun Ravi
# dpn.py

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipdb
import os
import tensorflow as tf





def dpn(X_train, y_train, activation=tf.nn.relu):

	x = tf.placeholder(tf.float32, shape=[X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]], name='x')
	y = tf.placeholder(tf.float32, shape=[y_train.shape[0], y_train.shape[1], y_train.shape[2], y_train.shape[3]], name='y')

	# conv1
	conv1 = tf.layers.conv2d(x, 64, (7,7), activation=activation)
	

	# conv2
	conv2 = tf.layers.conv2d(conv1, 64, (3,3), activation=activation)
	conv2 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(3,3), strides=2)
	for i in range(0, 3):
		conv2 = tf.layers.conv2d(conv2, 96, (1,1), activation=activation)
		conv2 = tf.layers.conv2d(conv1, 96, (3,3), activation=activation)
		conv2 = tf.layers.conv2d(conv2, 256, (1,1), activation=activation)

	# conv3
	for i in range(0, 4):
		conv2 = tf.layers.conv2d(conv2, 192, (1,1), activation=activation)
		conv2 = tf.layers.conv2d(conv1, 192, (3,3), activation=activation)
		conv2 = tf.layers.conv2d(conv2, 512, (1,1), activation=activation)

	# conv4
	for i in range(0, 20):
		conv3 = tf.layers.conv2d(conv2, 384, (1,1), activation=activation)
		conv3 = tf.layers.conv2d(conv1, 384, (3,3), activation=activation)
		conv3 = tf.layers.conv2d(conv2, 1024, (1,1), activation=activation)

	# conv5
	for i in range(0, 3):
		conv4 = tf.layers.conv2d(conv2, 768, (1,1), activation=activation)
		conv4 = tf.layers.conv2d(conv1, 768, (3,3), activation=activation)
		conv4 = tf.layers.conv2d(conv2, 2048, (1,1), activation=activation)

	final = tf.layers.conv2d(up_conv4, 2, 1, activation=tf.nn.softmax)

	ipdb.set_trace()



if __name__ == '__main__':

	X_image = np.zeros((1, 572, 572, 1))
	y_image = np.zeros((1, 572, 572, 1))

	dpn(X_image, y_image)