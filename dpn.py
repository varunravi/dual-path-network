# Varun Ravi
# dpn.py
# 6/28/18

import ipdb
import tensorflow as tf
import numpy as np
from keras.datasets import cifar100


EPOCHS = 1
BATCH_SIZE = 8
DATA = 'places365'
X_SHAPE = [None,32,32,3]
y_SHAPE = [None,10]
CARDINALITY = 32


def get_data(data):

	if data == 'cifar100':
		(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.cifar10.load_data()
		train_labels = one_hot(train_labels, (50000,10))
		eval_labels = one_hot(eval_labels, (50000,10))

	if data == 'cifar10':
		(train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.cifar10.load_data()
		train_labels = one_hot(train_labels, (50000,10))
		eval_labels = one_hot(eval_labels, (50000,10))

	if data == 'places365':
		train_data = np.load('/home/varun/datasets/places365/places365standard_easyformat/places365_standard/images_batch_20.npy')
		#train_data = np.concatenate((train_data, np.load('/home/varun/datasets/places365/places365standard_easyformat/places365_standard/images_batch_40.npy')))
		#train_data = np.concatenate((train_data, np.load('/home/varun/datasets/places365/places365standard_easyformat/places365_standard/images_batch_80.npy')))
		
		train_labels = np.load('/home/varun/datasets/places365/places365standard_easyformat/places365_standard/keys_batch_20.npy')
		#train_labels = np.load('/home/varun/datasets/places365/places365standard_easyformat/places365_standard/keys_batch_40.npy')
		#train_labels = np.load('/home/varun/datasets/places365/places365standard_easyformat/places365_standard/keys_batch_80.npy')
		#ipdb.set_trace()
		train_labels = one_hot(train_labels, (1901,365))
		eval_data = None
		eval_labels = None
		
	return train_data[:5], train_labels[:5], eval_data, eval_labels

def one_hot(labels, shape):
	one_hot= np.zeros(shape)
	pos=0

	for i in labels:
		one_hot[pos][i] = 1
		pos+=1

	return one_hot

def macro_block(inputs, filter_size, iterations, cardinality=CARDINALITY):
	
	inputs_G = []

	for i in range(cardinality):
		inputs_G.append(micro_block(inputs, filter_size, iterations))
	
	inputs_final = tf.concat(values=inputs_G, axis=3, name='concat')

	return inputs_final

def micro_block(inputs, final_filter, iters, padding='same'):

	small_filter = final_filter/(2+2/3)

	for i in range(iters):
		inputs = tf.layers.conv2d(inputs=inputs, filters=small_filter, kernel_size=(1,1), padding=padding)
		inputs = tf.layers.conv2d(inputs=inputs, filters=small_filter, kernel_size=(3,3), padding=padding)
		inputs = tf.layers.conv2d(inputs=inputs, filters=final_filter, kernel_size=(1,1), padding=padding)

	return inputs

def model(x,y):

	for d in ['/device:GPU:0', '/device:GPU:1']:
		with tf.device(d):

			conv2_G = []

			inputs = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(7,7), strides=(2,2), padding='same')
			layer = tf.layers.max_pooling2d(inputs=inputs, pool_size=(3,3), strides=2)

			layer = macro_block(inputs=layer, filter_size=256, iterations=3)
			res_layer = layer
			dense_layer = layer

			layer = macro_block(inputs=layer, filter_size=512, iterations=4)
			layer_r = layer[:,:,:,:res_layer.shape[3]]
			res_layer = tf.add(res_layer,layer_r)
			dense_layer = tf.concat([dense_layer,layer], axis=3)
			layer = tf.concat([res_layer,dense_layer], axis=3)

			layer = macro_block(inputs=layer, filter_size=1024, iterations=20)
			layer_r = layer[:,:,:,:res_layer.shape[3]]
			res_layer = tf.add(res_layer,layer_r)
			dense_layer = tf.concat([dense_layer,layer], axis=3)
			layer = tf.concat([res_layer,dense_layer], axis=3)

			layer = macro_block(inputs=layer, filter_size=2048, iterations=3)
			layer_r = layer[:,:,:,:res_layer.shape[3]]
			res_layer = tf.add(res_layer,layer_r)
			dense_layer = tf.concat([dense_layer,layer], axis=3)
			layer = tf.concat([res_layer,dense_layer], axis=3)

			layer = tf.layers.average_pooling2d(layer, (2,2), 2, padding='same')
			layer = tf.contrib.layers.flatten(layer)
			layer = tf.layers.dense(inputs=layer, units=1000, activation=tf.nn.relu)
			layer = tf.layers.dense(inputs=layer, units=100)
			#layer = tf.layers.dense(inputs=layer, units=100, activation=tf.nn.softmax)

	return layer

if __name__ == '__main__':

	# dimensions
	x = tf.placeholder(tf.float32, shape=X_SHAPE, name='x')
	y = tf.placeholder(tf.float32, shape=y_SHAPE, name='y')

	# data
	train_data, train_labels, eval_data, eval_labels = get_data(DATA)
	#ipdb.set_trace()
	# model, loss & activation functions
	model = model(x,y)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y))
	optimize = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

	# training
	init = tf.global_variables_initializer()	
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement=True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.7

	with tf.Session(config=config) as session:
		session.run(init)
		for epoch in range(0, EPOCHS):
			total_steps = int(len(train_data)/BATCH_SIZE)
			#total_steps=200
			for step in range(200):
				start = step*BATCH_SIZE
				end = start+BATCH_SIZE
				X_batch = train_data[start:end]
				y_batch = train_labels[start:end]
				
				_, training_loss = session.run([optimize, loss], feed_dict={x: X_batch, y: y_batch})
				
				print ("Steps %d/%d, Training Loss=%f" % (step+1, total_steps, training_loss))

			total_steps = int(len(eval_data)/BATCH_SIZE)
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

		
