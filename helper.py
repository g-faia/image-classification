from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import tarfile
from os.path import join

import tensorflow as tf
import numpy as np
from six.moves import urllib

from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.datasets import mnist


# The path of data directory
DATA_DIR = join(os.path.dirname(__file__), "data")
MNIST_DIR = join(DATA_DIR, 'MNIST_data')
CIFAR10_DIR = join(DATA_DIR, 'cifar-10-batches-py')


def _train_pre_process_image(image, img_size_cropped):
  """Pre-process the image in train phase."""
  # img_depth = int(image.shape[-1])
  
  # Randomly crop the input image.
  image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, 3])

  # Randomly flip the image horizontally.
  image = tf.image.random_flip_left_right(image)

  # Randomly adjust hue, contrast and saturation.
  image = tf.image.random_hue(image, max_delta=0.05)
  image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
  image = tf.image.random_brightness(image, max_delta=0.2)
  image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

  # Some of these functions may overflow and result in pixel
  # values beyond the [0, 1] range. It is unclear from the
  # documentation of TensorFlow 0.10.0rc0 whether this is
  # intended. A simple solution is to limit the range.

  # Limit the image pixels between [0, 1] in case of overflow.
  image = tf.minimum(image, 1.0)
  image = tf.maximum(image, 0.0)
  
  return image


def _test_pre_process_image(image, img_size_cropped):
  """Pre-process the image in test phase."""
  # Crop the input image around the centre so it is the same
  # size as images that are randomly cropped during training.
  return tf.image.resize_image_with_crop_or_pad(image, 
                                                target_height=img_size_cropped, 
                                                target_width=img_size_cropped) 


def pre_process_image(image, phase, img_size_cropped):
  """Pre-process the image according to `phase`."""
  return tf.cond(phase, 
                 lambda: _train_pre_process_image(image, img_size_cropped), 
                 lambda: _test_pre_process_image(image, img_size_cropped))


def pre_process_images(images, phase, img_size_cropped):
  """Pre-process the images according to `phase`."""
  with tf.name_scope('pre_process'):
    return tf.map_fn(lambda image: pre_process_image(image, phase, img_size_cropped), images)


def variable_summaries(var):
  """Attach summaries of var to TensorBoard"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_mean(var))
    tf.summary.scalar('min', tf.reduce_mean(var))
    tf.summary.histogram('histogram', var)


def randomly_sample(data, labels, batch_size=128):
  """Randomly sample `batch_size` samples from dataset."""
  n_samples = data.shape[0]
  selected_samples = np.random.choice(n_samples, batch_size)
  return data[selected_samples], labels[selected_samples]


def plot_first_9_images(images, labels, classes, interpolation='spline16'):
  """Plot first images in dataset to make true the correct of dataset."""
  n_samples = images.shape[0]
  if n_samples == 0: return

  fig, axes = plt.subplots(3, 3)
  fig.subplots_adjust(hspace=0.6, wspace=0.3)

  for i, ax in enumerate(axes.flat):
    ax.imshow(images[i, :, :, :], interpolation=interpolation)

    ax.set_xlabel(classes[labels[i]])
    ax.set_xticks([])
    ax.set_yticks([])

  plt.show()


def maybe_download(file_dir, data_url, filename=None, extract=False):
  """Download and extract model tar file."""
  if not os.path.exists(file_dir):
    os.makedirs(file_dir)
  if filename is None:
    filename = data_url.split('/')[-1]

  file_path = os.path.join(file_dir, filename)
  if not os.path.exists(file_path):
    # def _progress(count, block_size, total_size):
    #   sys.stdout.write('\r>> Downloading %s %.1f%%' 
    #     % (filename, float(count * block_size) / float(total_size) * 100.0))
    #   sys.stdout.flush()
    # file_path, _ = urllib.request.urlretrieve(data_url, file_path, _progress)
    temp_file_name, _ = urllib.request.urlretrieve(data_url)
    gfile.Copy(temp_file_name, file_path)
    with gfile.GFile(file_path) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')

  if extract:
    tarfile.open(file_path, 'r:gz').extractall(file_dir)
  # Do not delete the downloaded file.
  # os.remove(file_path)

  return file_path


def detect_dir_is_existed(dir):
  """Detect whether the directory existed."""
  if tf.gfile.Exists(dir):
    tf.gfile.DeleteRecursively(dir)
  else:
    tf.gfile.MakeDirs(dir)


def generate_batches(data, labels, batch_size=128):
  """Output the next batch of the dataset."""
  n_samples = len(data)

  shuffle_indices = np.random.permutation(n_samples)
  data, labels = data[shuffle_indices], labels[shuffle_indices]

  n_batches = n_samples // batch_size + 1
  
  def _generate_batch():

    for i in range(n_batches):
      data_batch = data[i * batch_size: (i + 1) *  batch_size]
      label_batch = labels[i * batch_size: (i + 1) *  batch_size]

      yield data_batch, label_batch

  batches = [(xt, yt) for xt, yt in _generate_batch()]
  return batches


def mnist_data_loader(one_hot=False, reshape=True):
  """Load MNIST dataset."""
  # Download the dataset if not exist.
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  DATA_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  file_list = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]
  file_list = [maybe_download(MNIST_DIR, DATA_URL + fil) for fil in file_list]

  with gfile.Open(file_list[0], 'rb') as f:
    train_data = mnist.extract_images(f) / 255

  with gfile.Open(file_list[1], 'rb') as f:
    train_labels = mnist.extract_labels(f, one_hot)

  with gfile.Open(file_list[2], 'rb') as f:
    test_data = mnist.extract_images(f) / 255

  with gfile.Open(file_list[3], 'rb') as f:
    test_labels = mnist.extract_labels(f, one_hot)

  # Convert the shape of image, if reshape
  # [n_samples, width, length, 1] ==> [n_samples, n_features]
  if reshape:
    assert train_data.shape[1:] == test_data.shape[1:]
    n_train, width, length, _ = train_data.shape
    n_test = test_data.shape[0]
    train_data = train_data.reshape(n_train, width * length)
    test_data = test_data.reshape(n_test, width * length)

  return train_data, train_labels, test_data, test_labels


def _cifar_unpickle(file):
  """Unpickle the cifar data files."""
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict


def cifar_classes(file_dir):
  """Return the classes of the cifar dataset."""
  dic = _cifar_unpickle(file_dir + "batches.meta")
  return dic[b'label_names']


def cifar_data_reshape(data):
  """The size of original image is (n_samples, 32 * 32 * 3), and the interval of values is 0~255.
  
  Reshape the size of data to fit the model's input tensor size. 
  Change the interval of values of image from 0~255 into 0~1.

  Return, size of tensor (n_samples, 32, 32, 3)
  """
  n_dims = len(data.shape)
  if n_dims == 1:
    return data.reshape(3, 32, 32).transpose([0, 2, 3, 1]) / 255
  else:
    n_samples = data.shape[0]
    return data.reshape(n_samples, 3, 32, 32).transpose([0, 2, 3, 1]) / 255 


def cifar_data_loader(n_train_files=5, n_test_files=1):
  """Read all data from file directory."""
  if not tf.gfile.Exists(CIFAR10_DIR):
    raise Exception("Data directory {0} is not existed.".format(CIFAR10_DIR))
  if n_train_files == 0 or n_test_files == 0:
    raise Exception("`n_train_files` and `n_test_files` cannot be zero.")

  train_data = []
  train_labels = []
  for i in range(n_train_files):
    train_batch = _cifar_unpickle(CIFAR10_DIR + '/data_batch_{0}'.format(i + 1))
    train_data.append(train_batch[b'data'])
    train_labels.append(train_batch[b'labels'])

  train_data = cifar_data_reshape(np.concatenate(train_data, axis=0))
  train_labels = np.concatenate(train_labels)

  test_batch = _cifar_unpickle(CIFAR10_DIR + '/test_batch')
  test_data = cifar_data_reshape(test_batch[b'data'])
  # Note: Convert `test_labels` type to np.ndarray
  test_labels = np.array(test_batch[b'labels'])

  return train_data, train_labels, test_data, test_labels
