# Varun Ravi


import tensorflow as tf


def create_iterator(filename, batch_size, num_epochs):

	with tf.Graph().as_default() as graph:
		
		filenames = tf.constant(filename, dtype=tf.string, shape=None)
		
		dataset = tf.data.TextLineDataset(filenames)
		dataset = dataset.shuffle(buffer_size=10000)
		dataset = dataset.batch(batch_size)
		dataset = dataset.repeat(num_epochs)
		
		iterator = dataset.make_initializable_iterator()
		#next_element = iterator.get_next()
	
	return graph, iterator


if __name__ == '__main__':

	train_batch_size = 100
	val_batch_size = 5
	testing_batch_size = None
	val_filenames = ["/home/varun/datasets/places365/places365standard_easyformat/places365_standard/val.txt"]
	train_filenames = ["/home/varun/datasets/places365/places365standard_easyformat/places365_standard/train.txt"]
	testing_filenames = ["/home/varun/datasets/places365/places365standard_easyformat/places365_standard/val.txt"]
	
	val_graph, val_iterator = create_iterator(val_filenames, batch_size = 1000, num_epochs = 2)

	# with tf.Graph().as_default() as graph:	
	# 	filenames = tf.placeholder(tf.string, shape=[None])
		
	# 	dataset = tf.data.TextLineDataset(filenames)
	# 	dataset = dataset.shuffle(buffer_size=10000)
	# 	dataset = dataset.batch(32)
	# 	dataset = dataset.repeat(2)
		
	# 	iterator = dataset.make_initializable_iterator()


	with tf.Session(graph=val_graph) as sess:

		sess.run(tf.global_variables_initializer())
		sess.run(val_iterator.initializer)
		val_f = sess.run(val_iterator.get_next())
		print(val_f.shape)




	# iterator = dataset.make_one_shot_iterator()
	# init = tf.global_variables_initializer()

	# with tf.Session() as sess:
		
	# 	sess.run(init)
	# 	print(sess.run(iterator.initializer, feed_dict={filenames: val_filenames}))
	# 	#print(sess.run(iterator.get_next()))
