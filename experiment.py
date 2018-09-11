from cnn_train import train
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import shuffle

import argparse
import datasets
import numpy as np
import random

import DCGAN.main

class Flags:
  def __init__(self):
    self.epoch = 25
    self.learning_rate = 0.0002
    self.beta1 = 0.5
    self.train_size = np.inf
    self.batch_size = 64
    self.dataset = 'bowecho'
    self.class_index = None
    self.checkpoint_dir = './checkpoint'
    self.sample_dir = './data/samples'
    self.s_dim = 1000
    self.z_dim = 100

def combine_data(X1, y1, X2, y2, amt=-1):
	if amt == -1:
		amt = X2.shape[0]
	i = random.sample(range(X2.shape[0]), amt)
	X1 = np.append(X1, X2[i], axis=0)
	y1 = np.append(y1, y2[i], axis=0)
	X1, y1 = shuffle(X1, y1)
	return X1, y1.astype(int)

def force_data_imbalance(X, y, urep_class, urep_ratio):
	samples_per_class = np.bincount(y)
	n_classes = np.max(y) + 1

	# compute average samples per class excluding the target underrepresented class
	avg_samples_per_class = (np.sum(samples_per_class) - samples_per_class[urep_class]) / (n_classes - 1)
	urep_n_samples = int(avg_samples_per_class / urep_ratio)
	n_remove = samples_per_class[urep_class] - urep_n_samples
	if n_remove < 0:
		print('Not enough samples in the underrepresented class to reach the desired underrepresentation ratio. '
		      'Need {} more samples. Aborting program ...'.format(-n_remove))
		exit(0)
	urep_class_index = np.where(y == urep_class)[0]
	remove_index = urep_class_index[: n_remove]
	X = np.delete(X, remove_index, axis=0)
	y = np.delete(y, remove_index, axis=0)
	return X, y

def test_data_imbalance(y, urep_class):
	n_classes = y.max() + 1
	samples_per_class = np.bincount(y)

	# compute average samples per class excluding the target underrepresented class
	avg_samples_per_class = (np.sum(samples_per_class) - samples_per_class[urep_class]) / (n_classes - 1)
	urep_n_samples = samples_per_class[urep_class]
	urep_ratio = avg_samples_per_class / urep_n_samples
	print('urep_ratio: {}'.format(urep_ratio))

def experiment(options):
	print('CNN Flags')
	print(options)
	dataset_name = options['dataset_name']
	urep_class = options['urep_class']
	urep_ratio = options['urep_ratio']
	load_gen = options['load_gen']
	epochs_gen = options['epochs_gen']
	r_gen = options['r_gen']
	use_validation_step = options['use_validation_step']
	train_size = options['train_size']
	test_size = options['test_size']
	n_splits = options['n_splits']

	X, y = datasets.load_data(dataset_name)
	y = y.astype(int)
	n_classes = y.max() + 1

	if urep_class is not None:
		test_data_imbalance(y, urep_class)

	if epochs_gen:
		flags = Flags()
		flags.epoch = epochs_gen
		flags.class_index = urep_class

	#eval = [0, np.zeros(n_classes), np.zeros(n_classes), 0, np.zeros(n_classes), np.zeros(n_classes)]
	if use_validation_step:
		sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size)
		train_index, test_index = next(sss.split(X, y))
		val_index = np.delete(np.arange(X.shape[0]), np.concatenate((train_index, test_index)))
		X_train, y_train = shuffle(X[train_index], y[train_index])
		X_val, y_val = shuffle(X[val_index], y[val_index])
		X_test, y_test = shuffle(X[test_index], y[test_index])
		if urep_class is not None and urep_ratio is not None:
			X_train, y_train = force_data_imbalance(X_train, y_train, urep_class, urep_ratio)
			test_data_imbalance(y_train, urep_class)
		if epochs_gen or load_gen:
			if epochs_gen:
				r_gen = r_gen if r_gen else 1
				flags.s_dim = r_gen * np.bincount(y_train)[urep_class]
				print('DCGAN Flags')
				print(vars(flags))
				X_gen, y_gen = DCGAN.main.train(X_train, y_train, flags)
			elif load_gen:
				X_gen, y_gen = datasets.load_data('generated')
			X_train, y_train = combine_data(X_train, y_train, X_gen, y_gen)
			test_data_imbalance(y_train, urep_class)
		eval = train(X_train, y_train, X_val, y_val, X_test, y_test, options)
	# else:
	# 	skf = StratifiedKFold(n_splits=n_splits)
	# 	for train_index, test_index in skf.split(X, y):
	# 		X_train, y_train = shuffle(X[train_index], y[train_index])
	# 		X_test, y_test = shuffle(X[test_index], y[test_index])
	# 		if urep_class is not None and urep_ratio is not None:
	# 			X_train, y_train = force_data_imbalance(X_train, y_train, urep_class, urep_ratio)
	# 			test_data_imbalance(y_train, urep_class)
	# 		if n_gen != 0:
	# 			X_train, y_train = combine_data(X_train, y_train, X_gen, y_gen, n_gen)
	# 			test_data_imbalance(y_train, urep_class)
	# 		split_eval = train(X_train, y_train, None, None, X_test, y_test, options)
	# 		for i in range(6):
	# 			eval[i] += split_eval[i] / n_splits

	for perf in eval:
		print(perf)

	return eval