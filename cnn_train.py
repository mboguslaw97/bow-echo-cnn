# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_sample_weight

import cnn_model
import datasets
import numpy as np
import tensorflow as tf
import os
import os.path as op


MODEL_DIRECTORY = op.join(os.getcwd(), 'model', 'model.ckpt')
LOGS_DIRECTORY = op.join(os.getcwd(), 'logs', 'train')
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

def evaluate(X_test, Y_test, tf_vars):
	sess, is_training, x, y_, y = tf_vars

	# Predict
	y_pred = sess.run(y, feed_dict={x: X_test, y_: Y_test, is_training: False})

	# One-hot decode
	y_true = np.argmax(Y_test, 1)
	y_pred = np.argmax(y_pred, 1)

	# Calculate accuracy, precision, and recall
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, average=None)
	recall = recall_score(y_true, y_pred, average=None)

	return accuracy, precision, recall

def train(X_train, y_train, X_val, y_val, X_test, y_test, options):

	# Extract options
	epochs = options['epochs']
	batch_size = options['batch_size']
	keep_prob = options['keep_prob']
	urep_class = options['urep_class']
	urep_weight = options['urep_weight']
	use_validation_step = options['use_validation_step']
	display_step = options['display_step']
	validation_step = options['validation_step']

	# Define some variables
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)
	train_size = X_train.shape[0]
	val_size = X_val.shape[0] if use_validation_step else 0
	test_size = X_test.shape[0]
	n_train_batches = train_size // batch_size
	n_test_batches = test_size // batch_size
	n_pixels = X_train.shape[1]
	n_classes = int(y_test.max()) + 1
	img_shape = [int(np.sqrt(n_pixels)), int(np.sqrt(n_pixels))] #temporary fix due to FPE

	# One hot encode labels
	Y_train = np.eye(n_classes)[y_train.astype(int)]
	Y_val = np.eye(n_classes)[y_val.astype(int)] if use_validation_step else None
	Y_test = np.eye(n_classes)[y_test.astype(int)]

	# Compute weights
	class_weights = np.ones(n_classes)
	class_weights[urep_class] = urep_weight
	sample_weights = class_weights[y_train]

	# if use_weighted_loss:
	# 	#sample_weights = compute_sample_weight('balanced', y_train)
	# 	#class_weights = train_size / (n_classes * np.bincount(y_train))
	# 	class_weights = np.ones(n_classes)
	# 	class_weights[urep_class] *= urep_weight
	# 	sample_weights = class_weights[y_train]
	# else:
	# 	sample_weights = np.ones(train_size)

	# Reset graph
	tf.reset_default_graph()

	# Boolean for MODE of train or test
	is_training = tf.placeholder(tf.bool, name='MODE')

	# tf graph input
	x = tf.placeholder(tf.float32, [None, n_pixels])
	y_ = tf.placeholder(tf.float32, [None, n_classes]) # truth
	w = tf.placeholder(tf.float32, [batch_size])
	y = cnn_model.CNN(x, img_shape, n_classes, keep_prob=keep_prob) # prediction

	# Get loss of model
	with tf.name_scope("LOSS"):
		loss = tf.losses.softmax_cross_entropy(y_, y, w)

	# Define optimizer
	with tf.name_scope("ADAM"):

		# Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
		batch = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(
			1e-4,               # Base learning rate. Originally 1e-4
			batch * batch_size, # Current index into the dataset.
			train_size,         # Decay step.
			0.95,               # Decay rate. Originally 0.95
			staircase=True
		)

		# Use simple momentum for the optimization.
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

	# Get accuracy of model
	with tf.name_scope("ACC"):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		# train_conf = tf.confusion_matrix(labels=tf.argmax(y_, 1), predictions=tf.argmax(y, 1), num_classes=n_classes)

	# Create a summary to monitor accuracy tensor
	tf.summary.scalar('accuracy', accuracy)

	# Merge all summaries into a single op
	merged_summary_op = tf.summary.merge_all()

	# Add ops to save and restore all the variables
	saver = tf.train.Saver()
	sess = tf.InteractiveSession(config=config)
	sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

	# op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

	# Save the maximum accuracy value for validation data
	max_acc = 0

	# Loop for epochs
	for epoch in range(epochs):

		# Random shuffling
		X_train, Y_train, sample_weights = shuffle(X_train, Y_train, sample_weights)

		# Loop for batches
		for i in range(n_train_batches):

			# Compute the offset of the current minibatch in the data.
			offset = (i * batch_size) % (train_size)
			batch_xs = X_train[offset:(offset + batch_size), :]
			batch_ys = Y_train[offset:(offset + batch_size), :]
			batch_sample_weights = sample_weights[offset:(offset + batch_size)]

			# Run optimization op (backprop), loss op (to get loss value), and summary nodes
			_, train_accuracy, summary = sess.run(
				[train_step, accuracy, merged_summary_op],
				feed_dict={x: batch_xs, y_: batch_ys, w: batch_sample_weights, is_training: True}
			)

			# Write logs at every iteimbalanced_class_ration
			summary_writer.add_summary(summary, epoch * n_train_batches + i)

		# Get accuracy for validation data
		if use_validation_step:

			# Calculate accuracy
			validation_accuracy = sess.run(accuracy, feed_dict={x: X_val, y_: Y_val, is_training: False})
			print('Epoch: {:03}, validation accuracy {:.5f}'.format(epoch + 1, validation_accuracy))

			# Save the current model if the maximum accuracy is updated
			if validation_accuracy > max_acc:
				max_acc = validation_accuracy
				save_path = saver.save(sess, MODEL_DIRECTORY)
				print('Model updated and saved in file: {}'.format(save_path))
		else:
			save_path = saver.save(sess, MODEL_DIRECTORY)

		print('Epoch: {:03}, training accuracy {:.5f}'.format(epoch + 1, train_accuracy))

	print("Optimization Finished!")

	# Restore variables from disk
	saver.restore(sess, MODEL_DIRECTORY)

	# Test accuracy of model
	tf_vars = (sess, is_training, x, y_, y)
	train_acc, train_prec, train_rec = evaluate(X_train, Y_train, tf_vars)
	test_acc, test_prec, test_rec = evaluate(X_test, Y_test, tf_vars)

	# Close sess for next training session
	sess.close()

	print("train accuracy for the stored model: {:.5f}".format(train_acc))
	for i in range(n_classes):
		print('train precision/recall for class {}: {:.5f}/{:.5f}'.format(i, train_prec[i], train_rec[i]))
	print()

	print("test accuracy for the stored model: {:.5f}".format(test_acc))
	for i in range(n_classes):
		print('test precision/recall for class {}: {:.5f}/{:.5f}'.format(i, test_prec[i], test_rec[i]))

	return train_acc, train_prec, train_rec, test_acc, test_prec, test_rec