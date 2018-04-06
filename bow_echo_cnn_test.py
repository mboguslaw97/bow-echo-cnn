# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

import bow_echo_data
import mnist_data
import cnn_model

def test(dataset, ratio=1):
    model_directory = 'model/model.ckpt'

    if dataset == 'bow_echo':
	    result = bow_echo_data.prepare_bow_echo_data(ratio)
    else:  # dataset == 'mnist':
	    result = mnist_data.prepare_MNIST_data()
    train_total_data, validation_data, validation_labels, test_data, test_labels, img_shape = result
    train_size = train_total_data.shape[0]
    num_labels = validation_labels.shape[1]
    num_pixels = validation_data.shape[1]
    print(np.bincount(np.argmax(test_labels, axis=1)))

    tf.reset_default_graph()

    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, num_pixels])
    y_ = tf.placeholder(tf.float32, [None, num_labels]) #answer
    y = cnn_model.CNN(x, img_shape, num_labels, is_training=is_training)

    # Add ops to save and restore all the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    # Restore variables from disk
    saver = tf.train.Saver()

    saver.restore(sess, model_directory)

    # =====================================================================

    train_data = train_total_data[:, :-num_labels]
    train_labels = train_total_data[:, -num_labels:]

    # Calculate accuracy for all mnist test images
    test_size = train_labels.shape[0]
    batch_size = test_size
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = train_data[offset:(offset + batch_size), :]
        batch_ys = train_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})

        correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))

        acc_buffer.append(np.sum(correct_prediction) / batch_size)

    train_acc = np.mean(acc_buffer)
    # =====================================================================

    # Calculate accuracy for all mnist test images
    test_size = validation_labels.shape[0]
    batch_size = test_size
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    # Loop over all batches
    for i in range(total_batch):
        # Compute the offset of the current minibatch in the data.
        offset = (i * batch_size) % (test_size)
        batch_xs = validation_data[offset:(offset + batch_size), :]
        batch_ys = validation_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})

        correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))

        acc_buffer.append(np.sum(correct_prediction) / batch_size)

    validation_acc = np.mean(acc_buffer)

    # =====================================================================

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

    test_acc = np.mean(acc_buffer)

    confusion = confusion_matrix(np.argmax(batch_ys, 1), np.argmax(y_final, 1))

    return train_acc, validation_acc, test_acc, confusion

if __name__ == '__main__':
    dataset = 'bow_echo'
    ratio = 25
    train_acc, validation_acc, test_acc, test_conf = test(dataset, ratio)
    nbe_test_acc = test_conf[0, 0] / np.sum(test_conf[0, :])
    be_test_acc = test_conf[1, 1] / np.sum(test_conf[1, :])
    result = 'train: {:.4}  validation: {:.4}  test:{:.4}\n'.format(train_acc, validation_acc, test_acc)
    result += 'not bow echo test accuracy: {:.4}\n'.format(nbe_test_acc)
    result += 'bow echo test accuracy: {:.4}\n'.format(be_test_acc)
    result += 'test confusion matrix:\n{}'.format(test_conf)
    print(result)

