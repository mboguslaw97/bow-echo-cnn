from PIL import Image
import glob
import h5py
import numpy as np
import os

def partition(total_bed, total_nbed, start, end):
	bed_amt = total_bed.shape[0]
	nbed_amt = total_nbed.shape[0]
	total_bed = total_bed[int(bed_amt * start) : int(bed_amt * end)]
	total_nbed = total_nbed[int(nbed_amt * start): int(nbed_amt * end)]
	total_data = np.concatenate((total_bed, total_nbed), axis=0)
	np.random.shuffle(total_data)
	return total_data

def proportionate_data(total_bow_echo_data, total_not_bow_echo_data, train_percent, validation_percent):
	np.random.shuffle(total_bow_echo_data)
	np.random.shuffle(total_not_bow_echo_data)

	train_total_data = partition(total_bow_echo_data, total_not_bow_echo_data, 0, train_percent)
	validation_total_data = partition(total_bow_echo_data, total_not_bow_echo_data, train_percent,
	                                  train_percent + validation_percent)
	test_total_data = partition(total_bow_echo_data, total_not_bow_echo_data, train_percent + validation_percent, 1)

	validation_data = validation_total_data[:, :-2]
	validation_labels = validation_total_data[:, -2:]
	test_data = test_total_data[:, :-2]
	test_labels = test_total_data[:, -2:]

	return train_total_data, validation_data, validation_labels, test_data, test_labels

def random_data(total_bow_echo_data, total_not_bow_echo_data, train_percent, validation_percent):
	total_data = np.concatenate((total_bow_echo_data, total_not_bow_echo_data), axis=0)
	np.random.shuffle(total_data)

	total_data_size = total_data.shape[0]
	train_size = int(train_percent * total_data_size)
	validation_size = int(validation_percent * total_data_size)
	test_size = int(validation_percent * total_data_size)

	train_index = 0
	validation_index = train_size
	test_index = train_size + validation_size
	end_index = test_index + test_size

	train_total_data = total_data[train_index: validation_index, :]
	validation_data = total_data[validation_index: test_index, 0: -2]
	validation_labels = total_data[validation_index: test_index, -2:]
	test_data = total_data[test_index: end_index, 0: -2]
	test_labels = total_data[test_index: end_index, -2:]

	return train_total_data, validation_data, validation_labels, test_data, test_labels

def extract_data(imgs_dir, max_img_count=float('inf')):
	paths = glob.glob(os.path.join(imgs_dir, '*.png'))
	img_count = int(min(len(paths), max_img_count))
	img = Image.open(paths[0]).convert('1')
	img_arr = np.array(img)
	pixel_count = np.prod(img_arr.shape)
	data = np.empty((img_count, pixel_count))
	for i in range(img_count):
		img = Image.open(paths[i]).convert('1')
		img_arr = np.array(img).reshape((1, pixel_count))
		data[i, :] = img_arr
	return data

def create_tensor(hdf_path, ratio=1):
	bow_echoes_dir = '/storage/home/meb6031/data/bow_echoes'
	not_bow_echoes_dir = '/storage/home/meb6031/data/not_bow_echoes'

	train_percent = .8
	validation_percent = .0

	bow_echoes = extract_data(bow_echoes_dir)
	not_bow_echoes = extract_data(not_bow_echoes_dir, bow_echoes.shape[0] // ratio)
	data = np.concatenate((bow_echoes, not_bow_echoes), axis=0)

	bel_shape = (bow_echoes.shape[0], 1)
	nbel_shape = (not_bow_echoes.shape[0], 1)

	bow_echo_labels = np.concatenate((np.zeros(bel_shape), np.ones(bel_shape)), axis=1)
	not_bow_echo_labels = np.concatenate((np.ones(nbel_shape), np.zeros(nbel_shape)), axis=1)
	labels = np.concatenate((bow_echo_labels, not_bow_echo_labels), axis=0)

	total_bow_echo_data = np.concatenate((bow_echoes, bow_echo_labels), axis=1)
	total_not_bow_echo_data = np.concatenate((not_bow_echoes, not_bow_echo_labels), axis=1)

	train_total_data, validation_data, validation_labels, test_data, test_labels = \
		proportionate_data(total_bow_echo_data, total_not_bow_echo_data, train_percent, validation_percent)

	# train_total_data, validation_data, validation_labels, test_data, test_labels = \
	# 	random_data(total_bow_echo_data, total_not_bow_echo_data, train_percent, validation_percent)

	with h5py.File(hdf_path, 'w') as hdf:
		hdf.create_dataset('train_total_data', data=train_total_data)
		hdf.create_dataset('validation_data', data=validation_data)
		hdf.create_dataset('validation_labels', data=validation_labels)
		hdf.create_dataset('test_data', data=test_data)
		hdf.create_dataset('test_labels', data=test_labels)

def prepare_bow_echo_data(ratio=1):
	if ratio == 1:
		hdf_path = '/storage/home/meb6031/data/bow_echo_data.h5'
	else:
		hdf_path = '/storage/home/meb6031/data/bed_{}to1.h5'.format(ratio)
	if (not os.path.isfile(hdf_path)):
		create_tensor(hdf_path, ratio)

	with h5py.File(hdf_path, 'r') as hdf:
		train_total_data = np.array(hdf.get('train_total_data'))
		validation_data = np.array(hdf.get('validation_data'))
		validation_labels = np.array(hdf.get('validation_labels'))
		test_data = np.array(hdf.get('test_data'))
		test_labels = np.array(hdf.get('test_labels'))
	img_shape = (55, 55)
	return train_total_data, validation_data, validation_labels, test_data, test_labels, img_shape

if __name__ == '__main__':
	train_total_data, validation_data, validation_labels, test_data, test_labels, img_shape\
		= prepare_bow_echo_data(ratio=35)
	print(np.bincount(np.argmax(test_labels, axis=1)))
	print(np.count_nonzero(train_total_data[:, -1]) / np.count_nonzero(train_total_data[:, -2]))
	print(np.count_nonzero(validation_labels[:, -1]) / np.count_nonzero(validation_labels[:, -2]))
	print(np.count_nonzero(test_labels[:, -1]) / np.count_nonzero(test_labels[:, -2]))