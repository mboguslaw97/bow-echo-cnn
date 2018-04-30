import argparse
from bow_echo_cnn_train import train
from bow_echo_cnn_test import test
import numpy as np
import os
import os.path as op
import pickle

def calc_class_accs(conf):
    # Calculate the accuracy of each class based on the confusion matrix
    class_accs = []
    for i in range(conf.shape[0]):
        class_accs.append(conf[i, i] / np.sum(conf[i, :]))
    return class_accs

def experiment(dataset='bow_echo', epochs=100, batch_size=150, keep_prob=0.5, use_weighted_loss=True, kfold=4,
               kfold_index=0, imbalanced_class_ratio=1, imbalanced_class_index=0, imbalanced_class_weight_mult=1):
    train(dataset, epochs, batch_size, keep_prob, use_weighted_loss, imbalanced_class_ratio,
          imbalanced_class_index, imbalanced_class_weight_mult, kfold, kfold_index)
    train_acc, train_conf, test_acc, test_conf = test(dataset, imbalanced_class_ratio, kfold, kfold_index)
    nbe_train_acc = train_conf[0, 0] / np.sum(train_conf[0, :])
    be_train_acc = train_conf[1, 1] / np.sum(train_conf[1, :])
    nbe_test_acc = test_conf[0, 0] / np.sum(test_conf[0, :])
    be_test_acc = test_conf[1, 1] / np.sum(test_conf[1, :])
    result = 'dataset={}, epochs={}, batch_size={}, keep_prob={}, use_weighted_loss={}\n'.format(dataset, epochs, batch_size, keep_prob, use_weighted_loss)
    result += 'imbalanced_class_ratio={}, imbalanced_class_index={}, imbalanced_class_weight_mult={}\n'.format(imbalanced_class_ratio, imbalanced_class_index, imbalanced_class_weight_mult)
    result += 'train accuracy: {:.4}\n'.format(train_acc)
    result += 'not bow echo train accuracy: {:.4}\n'.format(nbe_train_acc)
    result += 'bow echo train accuracy: {:.4}\n'.format(be_train_acc)
    result += 'train confusion matrix:\n{}\n'.format(train_conf)
    result += 'test accuracy:{:.4}\n'.format(test_acc)
    result += 'not bow echo test accuracy: {:.4}\n'.format(nbe_test_acc)
    result += 'bow echo test accuracy: {:.4}\n'.format(be_test_acc)
    result += 'test confusion matrix:\n{}\n\n'.format(test_conf)
    log_file = open('experiment1.log', 'a+')
    log_file.write(result)
    log_file.close()
    return train_acc, train_conf, test_acc, test_conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['bow_echo', 'mnist'], default='bow_echo')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batchsize', type=int, default=150)
    parser.add_argument('-k', '--keepprobability', type=float, default=0.5)
    parser.add_argument('-w', '--useweightedloss', type=str.lower, choices=['t', 'true', 'f', 'false'], default='t')
    parser.add_argument('-r', '--imbalancedclassratio', type=float, default=1,
                        help='ratio of balanced dataset size to imbalanced dataset size as a decimal')
    parser.add_argument('-i', '--imbalancedclassindex', type=int, default=0,
                        help='index of class to be made imbalanced with [ratio]')
    parser.add_argument('-m', '--imbalancedclassweightmultiplier', type=float, default=1,
                        help='value to scale imbalanced class weight')
    parser.add_argument('-f','--kfold',type=int,default=4, help='Number of folds for cross validation. Default 4, use 1 for no cross validation')
    args = parser.parse_args()

    use_weighted_loss = args.useweightedloss in ('t', 'true')
    kfold = args.kfold
    ratio = args.imbalancedclassratio
    mult = args.imbalancedclassweightmultiplier
    kfold_index = 0

    #experiment(args.dataset, args.epochs, args.batch_size, args.keep_probability, use_weighted_loss, args.imbalanced_class_ratio,
    #           args.imbalanced_class_index, args.imbalanced_class_weight_multiplier, kfold, kfold_index)

    train_accs_map = {}
    test_accs_map = {}
    avg_train_acc = 0
    avg_train_conf = 0
    avg_test_acc = 0
    avg_test_conf = 0
    for i in range(kfold):
        train_acc, train_conf, test_acc, test_conf = experiment(dataset=args.dataset,epochs=args.epochs,batch_size=args.batchsize,
                                                                keep_prob=args.keepprobability, use_weighted_loss=args.useweightedloss,
                                                                imbalanced_class_ratio=ratio, imbalanced_class_index=args.imbalancedclassindex,
                                                                imbalanced_class_weight_mult=mult, kfold=kfold, kfold_index=i)
        avg_train_acc += train_acc / kfold
        avg_train_conf += train_conf / kfold
        avg_test_acc += test_acc / kfold
        avg_test_conf += test_conf / kfold
    train_accs = [avg_train_acc] + calc_class_accs(avg_train_conf)
    test_accs = [avg_test_acc] + calc_class_accs(avg_test_conf)
    train_accs_map[(ratio, mult)] = train_accs
    test_accs_map[(ratio, mult)] = test_accs

    # abs_dir = '/storage/home/meb6031/bow_echo_cnn/plots/run1/'  # aci
    abs_dir = op.join(os.getcwd(),'plots','run2')
    pickle.dump(train_accs_map, open(op.join(abs_dir, 'train_accs_map.p'), "wb"))
    pickle.dump(test_accs_map, open(op.join(abs_dir, 'test_accs_map.p'), "wb"))




