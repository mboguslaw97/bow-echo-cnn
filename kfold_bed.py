from PIL import Image
import glob
import h5py
import numpy as np
import os
import os.path as op

def create_kfold_datasets(total_data, k):
    fold_size = total_data.shape[0] // k
    train_total_data_sets = []
    test_data_sets = []
    test_label_sets = []
    for i in range(k):
        train_total_data = np.concatenate((total_data[ : i * fold_size], total_data[(i + 1) * fold_size : ]), axis=0)
        test_total_data = total_data[i * fold_size : (i + 1) * fold_size]
        test_data = test_total_data[:, : -2]
        test_labels = test_total_data[:, -2 : ]
        train_total_data_sets.append(train_total_data)
        test_data_sets.append(test_data)
        test_label_sets.append(test_labels)
    return train_total_data_sets, test_data_sets, test_label_sets

def evenly_distribute(total_bow_echo_data, total_not_bow_echo_data, k):
    np.random.shuffle(total_bow_echo_data)
    np.random.shuffle(total_not_bow_echo_data)
    bed_amt = total_bow_echo_data.shape[0]
    nbed_amt = total_not_bow_echo_data.shape[0]
    folds = []
    for i in range(k):
        nbe_start = i * nbed_amt // k
        nbe_end = (i + 1) * nbed_amt // k
        not_bow_echo_data_fold = total_not_bow_echo_data[nbe_start: nbe_end]
        be_start = i * bed_amt // k
        be_end = (i + 1) * bed_amt // k
        bow_echo_data_fold = total_bow_echo_data[be_start : be_end]
        fold = np.concatenate((bow_echo_data_fold, not_bow_echo_data_fold), axis=0)
        np.random.shuffle(fold)
        folds.append(fold)
    total_data = np.concatenate(folds, axis=0)
    return total_data

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

def create_tensor(hdf_path, ratio=1, k=4):
    # Hard Coding
    #bow_echoes_dir = r'C:\Users\MBOGU\Documents\Research\bow_echo_cnn\data\bow_echoes' #local
    #not_bow_echoes_dir = r'C:\Users\MBOGU\Documents\Research\bow_echo_cnn\data\bow_echoes' #local
    #bow_echoes_dir = '/storage/home/meb6031/data/bow_echoes' #aci
    #not_bow_echoes_dir = '/storage/home/meb6031/data/not_bow_echoes' #aci
    #bow_echoes_dir = '/home/meb6031/data/bow_echoes' #galois
    #not_bow_echoes_dir = '/home/meb6031/data/not_bow_echoes' #galois
    '''TODO: Change the code in order to include all possible datasets'''
    bow_echoes_dir = op.join(os.getcwd(),'data','bow_echoes')
    not_bow_echoes_dir = op.join(os.getcwd(),'data','not_bow_echoes')

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

    total_data = evenly_distribute(total_bow_echo_data, total_not_bow_echo_data, k)

    train_total_data_sets, test_data_sets, test_label_sets = create_kfold_datasets(total_data, k)

    with h5py.File(hdf_path, 'w') as hdf:
        hdf.create_dataset('train_total_data_sets', data=train_total_data_sets)
        hdf.create_dataset('test_data_sets', data=test_data_sets)
        hdf.create_dataset('test_label_sets', data=test_label_sets)

def prepare_kfold_bed(ratio=1, k=4, index=0):
    img_shape = (55, 55)
    #hdf_path = r'C:\Users\MBOGU\Documents\Research\bow_echo_cnn\data\k{}fold_bed_{}to1.h5'.format(k, ratio) #local
    hdf_path = '/storage/home/meb6031/data/k{}fold_bed_{}to1.h5'.format(k, ratio) #aci
    #hdf_path = '/home/meb6031/data/k{}fold_bed_{}to1.h5'.format(k, ratio) #galois
    if (not os.path.isfile(hdf_path)):
        create_tensor(hdf_path, ratio, k)

    with h5py.File(hdf_path, 'r') as hdf:
        train_total_data_sets = np.array(hdf.get('train_total_data_sets'))
        test_data_sets = np.array(hdf.get('test_data_sets'))
        test_label_sets = np.array(hdf.get('test_label_sets'))

    return train_total_data_sets[index], test_data_sets[index], test_label_sets[index], img_shape

if __name__ == '__main__':
    k = 4
    ratio = 1
    for i in range(k):
        train_total_data, test_data, test_labels, img_shape = prepare_kfold_bed(ratio, k, i)
        print(np.count_nonzero(train_total_data[:, -2]), np.count_nonzero(train_total_data[:, -1]))