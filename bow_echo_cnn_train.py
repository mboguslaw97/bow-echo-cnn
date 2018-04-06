# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import bow_echo_cnn_test
import bow_echo_data
import mnist_data
import cnn_model

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

def train(dataset='bow_echo', epochs=100, batch_size=150, keep_prob=0.5, use_weighted_loss=True,
	      imbalanced_class_ratio=1, imbalanced_class_index=0, imbalanced_class_weight_mult=1):
	# Prepare data
	if dataset == 'bow_echo':
		#epochs = 100
		display_step = 10
		validation_step = 50
		result = bow_echo_data.prepare_bow_echo_data(imbalanced_class_ratio)
	else: #dataset == 'mnist':
		#epochs = 20
		display_step = 100
		validation_step = 500
		result = mnist_data.prepare_MNIST_data()
	train_total_data, validation_data, validation_labels, test_data, test_labels, img_shape = result
	train_size = train_total_data.shape[0]
	num_pixels = validation_data.shape[1]
	num_labels = validation_labels.shape[1]

	if use_weighted_loss:
		#extract training labels
		train_labels = train_total_data[:, -num_labels:]
		#convert one hot labels to integers
		train_int_labels = np.argmax(train_labels, axis=1)
		#count the amount of data per class
		train_label_bin = np.bincount(train_int_labels, minlength=num_labels)
		#compute weights inversely proportional to amount of data per class
		class_weights = (train_size / num_labels / train_label_bin)
		#multiply underrepresented class by a constant
		class_weights[imbalanced_class_index] *= imbalanced_class_weight_mult
		#create data weights from class weights
		train_data_weights = class_weights[train_int_labels]
		#calculate pos_weight used for a different loss function
		pos_weight = train_label_bin[0] / train_label_bin[1] / imbalanced_class_weight_mult
	else:
		train_data_weights = np.ones(train_size)
		pos_weight = 1

	tf.reset_default_graph()

	# Boolean for MODE of train or test
	is_training = tf.placeholder(tf.bool, name='MODE')

	# tf Graph input
	x = tf.placeholder(tf.float32, [None, num_pixels])
	y_ = tf.placeholder(tf.float32, [None, num_labels]) #answer
	w = tf.placeholder(tf.float32, [batch_size])

	# Predict
	y = cnn_model.CNN(x, img_shape, num_labels, keep_prob=keep_prob)

	# Get loss of model
	with tf.name_scope("LOSS"):

		#loss with data weights
		#loss = tf.losses.softmax_cross_entropy(y_, y, w)

		#same loss as above, but broken down
		#losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
		#losses = tf.multiply(losses, w)
		#loss = tf.reduce_mean(losses)

		#loss with pos_weight
		losses = tf.nn.weighted_cross_entropy_with_logits(y_, y, pos_weight)
		loss = tf.reduce_mean(losses)

	# Create a summary to monitor loss tensor
	tf.summary.scalar('loss', loss)

	# Define optimizer
	with tf.name_scope("ADAM"):
		# Optimizer: set up a variable that's incremented once per batch and
		# controls the learning rate decay.
		batch = tf.Variable(0)

		learning_rate = tf.train.exponential_decay(
			1e-4,  # Base learning rate. Originally 1e-4
			batch * batch_size,  # Current index into the dataset.
			train_size,  # Decay step.
			0.95,  # Decay rate. Originally 0.95
			staircase=True)
		# Use simple momentum for the optimization.
		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

	# Create a summary to monitor learning_rate tensor
	tf.summary.scalar('learning_rate', learning_rate)

	# Get accuracy of model
	with tf.name_scope("ACC"):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Create a summary to monitor accuracy tensor
	tf.summary.scalar('acc', accuracy)

	# Merge all summaries into a single op
	merged_summary_op = tf.summary.merge_all()

	# Add ops to save and restore all the variables
	saver = tf.train.Saver()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

	# Training cycle
	total_batch = int(train_size / batch_size)

	# op to write logs to Tensorboard
	summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

	# Save the maximum accuracy value for validation data
	max_acc = 0.

	# Loop for epoch
	for epoch in range(epochs):

		# Random shuffling
		np.random.shuffle(train_total_data)
		train_data_ = train_total_data[:, :-num_labels]
		train_labels_ = train_total_data[:, -num_labels:]

		# Loop over all batches
		for i in range(total_batch):

			# Compute the offset of the current minibatch in the data.
			offset = (i * batch_size) % (train_size)
			batch_xs = train_data_[offset:(offset + batch_size), :]
			batch_ys = train_labels_[offset:(offset + batch_size), :]
			batch_data_weights = train_data_weights[offset:(offset + batch_size)]
			# Run optimization op (backprop), loss op (to get loss value)
			# and summary nodes
			_, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op],
				feed_dict={x: batch_xs, y_: batch_ys, w: batch_data_weights, is_training: True})
			# Write logs at every iteimbalanced_class_ration
			summary_writer.add_summary(summary, epoch * total_batch + i)

			# Display logs
			if i % display_step == 0:
				print("Epoch:", '%04d,' % (epoch + 1),
				"batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

			# Get accuracy for validation data
			if i % validation_step == 0:
				# Calculate accuracy
				validation_accuracy = sess.run(accuracy,
				feed_dict={x: validation_data, y_: validation_labels, is_training: False})

				print("Epoch:", '%04d,' % (epoch + 1),
				"batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

			# Save the current model if the maximum accuracy is updated
			if validation_accuracy > max_acc:
				max_acc = validation_accuracy
				save_path = saver.save(sess, MODEL_DIRECTORY)
				print("Model updated and saved in file: %s" % save_path)

	print("Optimization Finished!")

	# Restore variables from disk
	saver.restore(sess, MODEL_DIRECTORY)

	# Calculate accuracy for all mnist test images
	test_size = test_labels.shape[0]
	batch_size = test_size
	total_batch = int(test_size / batch_size)

	acc_buffer = []

	# Loop over all batches
	for i in range(total_batch):
		# Compute the offset of the current minibatch in the data.
		offset = (i * batch_size) % (test_size)
		batch_xs = test_data[offset:(offset + batch_size), :]
		batch_ys = test_labels[offset:(offset + batch_size), :]

		y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
		correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))
		acc_buffer.append(np.sum(correct_prediction) / batch_size)

	print("test accuracy for the stored model: %g" % np.mean(acc_buffer))

if __name__ == '__main__':
	train(dataset='bow_echo', imbalanced_class_ratio=25)