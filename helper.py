import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from six.moves import urllib
import os
import sys
import tarfile


def weight_variable(shape, name='weights', mean=0, stddev=0.1):
  """Randomly generate values follow a normal distribution.
  variable ~ N(mean, stddev)"""
  return tf.Variable(tf.truncated_normal(
    shape=shape, mean=mean, stddev=stddev, name=name))


def bias_variable(shape, name='bias', constant=0.1):
  """Generate constants which equal to constant."""
  return tf.Variable(tf.constant(value=constant, shape=shape, name=name))


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


def conv2d_layer(
  inputs, layer_name, in_channels, out_channels, 
  filters_size, filters_mean=0, filters_stddev=1e-2, 
  biases_constant=0.1, strides=[1, 1, 1, 1], padding='SAME', act=tf.nn.relu
  ):
  """Convolution Neural Network layer."""
  with tf.name_scope(layer_name):
    filters = weight_variable(
      shape=[filters_size[0], filters_size[1], in_channels, out_channels], 
      mean=filters_mean, stddev=filters_stddev, name='filters')    
    biases = bias_variable(
      shape=[out_channels], constant=biases_constant, name='bias')
    preactivate = tf.nn.conv2d(
      input=inputs, filter=filters, strides=strides, padding=padding)
    activations = act(preactivate + biases, name='activation')

  return activations


def max_pool_layer(
  inputs, layer_name, k_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
  padding='SAME'
  ):
  """Max pooling layer"""
  with tf.name_scope(layer_name):
    return tf.nn.max_pool(
      value=inputs, ksize=k_size, strides=strides, padding=padding)


def nn_layer(
  input_tensor, input_dim, output_dim, layer_name, 
  weights_mean=0, weights_stddev=1e-2, biases_constant=0.1, act=tf.nn.relu
  ):
  """Neural Network layer."""
  with tf.name_scope(layer_name):
    weights = weight_variable(
      shape=[input_dim, output_dim], 
      mean=weights_mean, stddev=weights_stddev, name='weights')
    biases = bias_variable(
      shape=[output_dim], constant=biases_constant, name="bias")
    preactivate = tf.matmul(input_tensor, weights) + biases
    activations = act(preactivate, name='activation')
  
  return activations


def randomly_sample(data, labels, batch_size=128):
  """Randomly sample `batch_size` samples from dataset."""
  n_samples = data.shape[0]
  selected_samples = np.random.choice(n_samples, batch_size)
  return data[selected_samples], labels[selected_samples]

########################################################################
# Common image utils
########################################################################

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
  return tf.image.resize_image_with_crop_or_pad(
    image, target_height=img_size_cropped, target_width=img_size_cropped) 


def pre_process_image(image, phase, img_size_cropped):
  """Pre-process the image according to `phase`."""
  return tf.cond(phase, 
    lambda: _train_pre_process_image(image, img_size_cropped), 
    lambda: _test_pre_process_image(image, img_size_cropped))


def pre_process_images(images, phase, img_size_cropped):
  """Pre-process the images according to `phase`."""
  with tf.name_scope('pre_process'):
    return tf.map_fn(lambda image: pre_process_image(
      image, phase, img_size_cropped), images)


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

########################################################################
# cifar-10 utils
########################################################################

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
  """The size of data imported from `file_dir` is 
  (n_samples, 32 * 32 * 3), And, the interval of values is 0~255.
  
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


def cifar_read_data(file_dir, n_train_files, n_test_files):
  """Read all data from file directory."""
  if not tf.gfile.Exists(file_dir):
    raise Exception("Data directory {0} is not existed.".format(file_dir))
  if n_train_files == 0 or n_test_files == 0:
    raise Exception("`n_train_files` and `n_test_files` cannot be zero.")

  train_data = []
  train_labels = []
  for i in range(n_train_files):
    train_batch = _cifar_unpickle(file_dir + 'data_batch_{0}'.format(i + 1))
    train_data.append(train_batch[b'data'])
    train_labels.append(train_batch[b'labels'])

  train_data = cifar_data_reshape(np.concatenate(train_data, axis=0))
  train_labels = np.concatenate(train_labels)

  test_batch = _cifar_unpickle(file_dir + 'test_batch')
  test_data = cifar_data_reshape(test_batch[b'data'])
  # Note: Convert `test_labels` type to np.ndarray
  test_labels = np.array(test_batch[b'labels'])

  return train_data, train_labels, test_data, test_labels

########################################################################
# inception model
########################################################################

def maybe_download_and_extract(model_dir, data_url, filename=None):
  """Download and extract model tar file."""
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  if filename is None:
    filename = data_url.split('/')[-1]

  file_path = os.path.join(model_dir, filename)
  if not os.path.exists(file_path):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' 
        % (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    file_path, _ = urllib.request.urlretrieve(data_url, file_path, _progress)
    print()
    statinfo = os.stat(file_path)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(file_path, 'r:gz').extractall(model_dir)
  # Do not delete the downloaded file.
  # os.remove(file_path)


def detect_dir_is_existed(dir):
  """Detect whether the directory existed."""
  if tf.gfile.Exists(dir):
    tf.gfile.DeleteRecursively(dir)
  else:
    tf.gfile.MakeDirs(dir)
