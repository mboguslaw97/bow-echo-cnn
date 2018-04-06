import argparse
from bow_echo_cnn_train import train
from bow_echo_cnn_test import test
import numpy as np

def log(text):
	log_file = open('run_and_log.log', 'a+')
	log_file.write(text)
	log_file.close()

def experiment(dataset='bow_echo', epochs=100, batch_size=150, keep_prob=0.5, use_weighted_loss=True,
	           imbalanced_class_ratio=1, imbalanced_class_index=0, imbalanced_class_weight_mult=1):
	train(dataset, epochs, batch_size, keep_prob, use_weighted_loss,
		  imbalanced_class_ratio, imbalanced_class_index, imbalanced_class_weight_mult)
	train_acc, validation_acc, test_acc, test_conf = test(dataset, imbalanced_class_ratio)
	nbe_test_acc = test_conf[0, 0] / np.sum(test_conf[0, :])
	be_test_acc = test_conf[1, 1] / np.sum(test_conf[1, :])
	result = 'dataset={}, epochs={}, batch_size={}, keep_prob={}, use_weighted_loss={}\n'.format(dataset, epochs, batch_size, keep_prob, use_weighted_loss)
	result += 'imbalanced_class_ratio={}, imbalanced_class_index={}, imbalanced_class_weight_mult={}\n'.format(imbalanced_class_ratio, imbalanced_class_index, imbalanced_class_weight_mult)
	result += 'train: {:.4}  validation: {:.4}  test:{:.4}\n'.format(train_acc, validation_acc, test_acc)
	result += 'not bow echo test accuracy: {:.4}\n'.format(nbe_test_acc)
	result += 'bow echo test accuracy: {:.4}\n'.format(be_test_acc)
	result += 'test confusion matrix:\n{}\n\n'.format(test_conf)
	log(result)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str, choices=['bow_echo', 'mnist'], default='bow_echo')
	parser.add_argument('-e', '--epochs', type=int, default=100)
	parser.add_argument('-b', '--batch-size', type=int, default=150)
	parser.add_argument('-k', '--keep-probability', type=float, default=0.5)
	parser.add_argument('-w', '--use-weighted-loss', type=str.lower, choices=['t', 'true', 'f', 'false'], default='t')
	parser.add_argument('-r', '--imbalanced-class-ratio', type=float, default=1,
	                    help='ratio of balanced dataset size to imbalanced dataset size as a decimal')
	parser.add_argument('-i', '--imbalanced-class-index', type=int, default=0,
	                    help='index of class to be made imbalanced with [ratio]')
	parser.add_argument('-m', '--imbalanced-class-weight-multiplier', type=float, default=1,
	                    help='value to scale imbalanced class weight')
	args = parser.parse_args()

	use_weighted_loss = args.use_weighted_loss in ('t', 'true')

	#experiment(args.dataset, args.epochs, args.batch_size, args.keep_probability, use_weighted_loss,
	#           args.imbalanced_class_ratio, args.imbalanced_class_index, args.imbalanced_class_weight_multiplier)

	for ratio in range(10, 45, 5):
		for mult in [.25, .5, 1, 2, 4]:
			experiment(imbalanced_class_ratio=ratio, imbalanced_class_weight_mult=mult)

	# experiment(imbalanced_class_ratio=30, imbalanced_class_weight_mult=1000)

