import numpy as np
import os
import scipy.misc
import shutil

from .model import DCGAN
from .utils import pp, generate, to_json, show_all_variables

import tensorflow as tf

class Flags:
  def __init__(self):
    self.epoch = 400
    self.learning_rate = 0.0002
    self.beta1 = 0.5
    self.train_size = np.inf
    self.batch_size = 64
    self.dataset = 'bowecho'
    self.class_index = None
    self.checkpoint_dir = './checkpoint'
    self.sample_dir = './samples'
    self.s_dim = 1000
    self.z_dim = 100

def train(X, y, FLAGS):

  if os.path.exists(FLAGS.checkpoint_dir):
    shutil.rmtree(FLAGS.checkpoint_dir)
  if os.path.exists(FLAGS.sample_dir):
    shutil.rmtree(FLAGS.sample_dir)
  os.makedirs(FLAGS.checkpoint_dir)
  os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess, X, y,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        z_dim=FLAGS.z_dim,
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    dcgan.train(FLAGS)
    X, y = generate(sess, dcgan, FLAGS)
    return X, y

if __name__ == '__main__':
  import datasets
  X, y = datasets.load_bowecho()
  flags = Flags()
  flags.epoch = 200
  flags.class_index = 0
  train(X, y, flags)

