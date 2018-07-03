
import tensorflow as tf
import ipdb

def macro_block(inputs, filter_size, iterations, cardinality=32):
	
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

def model(x_shape, y_shape):

	conv2_G = []

	X = tf.placeholder(tf.float32, shape=x_shape, name='X')
	y = tf.placeholder(tf.float32, y_shape, 'y')

	inputs = tf.layers.conv2d(inputs=X, filters=64, kernel_size=(7,7), strides=(2,2), padding='same')
	layer = tf.layers.max_pooling2d(inputs=inputs, pool_size=(3,3), strides=2)
	

	with tf.device('/gpu:0'):
		layer = macro_block(inputs=layer, filter_size=256, iterations=1)
		res_layer = layer
		dense_layer = layer

		layer = macro_block(inputs=layer, filter_size=512, iterations=1)
		layer_r = layer[:,:,:,:res_layer.shape[3]]
		res_layer = tf.add(res_layer,layer_r)
		dense_layer = tf.concat([dense_layer,layer], axis=3)
		layer = tf.concat([res_layer,dense_layer], axis=3)

		layer = macro_block(inputs=layer, filter_size=1024, iterations=1)
		layer_r = layer[:,:,:,:res_layer.shape[3]]
		res_layer = tf.add(res_layer,layer_r)
		dense_layer = tf.concat([dense_layer,layer], axis=3)
		layer = tf.concat([res_layer,dense_layer], axis=3)

	with tf.device('/gpu:1'):
		layer = macro_block(inputs=layer, filter_size=2048, iterations=1)
		layer_r = layer[:,:,:,:res_layer.shape[3]]
		res_layer = tf.add(res_layer,layer_r)
		dense_layer = tf.concat([dense_layer,layer], axis=3)
		layer = tf.concat([res_layer,dense_layer], axis=3)

		layer = tf.layers.average_pooling2d(layer, (2,2), 2, padding='same')
		layer = tf.contrib.layers.flatten(layer)
		layer = tf.layers.dense(inputs=layer, units=1000, activation=tf.nn.relu)
		layer = tf.layers.dense(inputs=layer, units=10, activation=tf.nn.softmax)


	return layer

	#dense_layer = tf.concat([dense_layer, conv3], axis=3)

	



if __name__ == '__main__':

	X_shape = [1,32,32,3]
	y_shape = [1,10]

	model(X_shape, y_shape)