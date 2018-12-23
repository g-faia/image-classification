import sys
import os

# If the scripts run on the tf-gpu, delete this line.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('..')

import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
from helper import *


def batch_norm_relu(inputs, is_training):
  """Performs a batch normalization followed by a ReLU."""
  inputs = tf.layers.batch_normalization(inputs=inputs, training=is_training)
  return tf.nn.relu(inputs)


def fixed_padding(inputs, kernel_size):
  """
  Pads the input along the spatial dimensions independently of input size.
  """

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  padded_inputs = tf.pad(inputs, [[0, 0], 
    [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding."""
  if strides > 1:
      inputs = fixed_padding(inputs, kernel_size)
  
  return tf.layers.conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'), use_bias=False, 
    kernel_initializer=tf.variance_scaling_initializer())


def building_block(inputs, filters, projection_shortcut, strides, is_training):
  """
  Standard building block for residual networks with BN before convolutions.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training)
  
  if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)
  
  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=strides)
  
  inputs = batch_norm_relu(inputs, is_training)
  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=1)
  
  return inputs + shortcut


def bottlenect_block(inputs, filters, projection_shortcut, strides):
  """
  Bottleneck block variant for residual networks with BN before convolutions.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, phase)
  
  if projection_shortcut is not None:
      shortcut = projection_shortcut(inputs)
      
  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=1, strides=1)
  
  inputs = batch_norm_relu(inputs, phase)
  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=filters, kernel_size=3, strides=strides)
  
  inputs = batch_norm_relu(inputs, phase)
  inputs = conv2d_fixed_padding(
    inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
  
  return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name):
  """Creates one layer of blocks for the ResNet model."""
  filters_out = 4 * filters if block_fn is bottlenect_block else filters
  
  def projection_shortcut(inputs):
      return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)
  
  # In the first block, add direct projection shortcut
  # If strides be set as 2 to downsample.
  inputs = block_fn(inputs, filters, projection_shortcut, strides, is_training)
  
  for _ in range(1, blocks):
      inputs = block_fn(inputs, filters, None, 1, is_training)
  
  return tf.identity(inputs, name)


class Resnet(object):

  def __init__(self, 
    image_size, img_depth, padded_size, cropped_size, num_classes, 
    weight_decay, init_lr, decay_steps, decay_rate):

    self.num_classes = num_classes
    self.weight_decay = weight_decay

    self.num_blocks = (image_size - 2) // 6
    self.inputs = tf.placeholder(
      tf.float32, [None, image_size, image_size, img_depth], name='inputs')
    
    # pad the original image to padded_size * padded_size
    pad_left = pad_up = (padded_size - image_size) // 2
    pad_right = pad_down = padded_size - image_size - pad_left
    self.padded_images = tf.pad(
      self.inputs, [[0, 0], [pad_left, pad_right], [pad_up, pad_down], [0, 0]])
    self.phase = tf.placeholder(dtype=tf.bool, shape=(), name='phase')
    with tf.device('/cpu:0'):
      self.distorted_images = pre_process_images(
        images=self.padded_images, phase=self.phase, 
        img_size_cropped=cropped_size)
    
    # use the distorted images as the input of model.
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.onehot_labels = tf.one_hot(
      indices=self.labels, depth=self.num_classes)
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(
      init_lr, global_step=self.global_step, 
      decay_steps=decay_steps, decay_rate=decay_rate)

    self.model()
    self.train_op()

  def model(self):

    initial_conv = conv2d_fixed_padding(
      inputs=self.distorted_images, filters=16, kernel_size=3, strides=1)
    initial_conv = tf.identity(initial_conv, 'initial_conv')

    block_layers1 = block_layer(
      inputs=initial_conv, filters=16, block_fn=building_block, 
      blocks=self.num_blocks, is_training=self.phase, strides=1, 
      name="block_layer1")
    block_layers2 = block_layer(
      inputs=block_layers1, filters=32, block_fn=building_block, 
      blocks=self.num_blocks, is_training=self.phase, strides=2, 
      name='block_layer2')
    block_layers3 = block_layer(
      inputs=block_layers2, filters=64, block_fn=building_block, 
      blocks=self.num_blocks, is_training=self.phase, strides=2, 
      name='block_layer3')
    
    final_norm = batch_norm_relu(block_layers3, self.phase)
    final_avg_pool = tf.layers.average_pooling2d(
      inputs=block_layers3, pool_size=8, strides=1, padding='VALID')

    final_avg_pool = tf.identity(final_avg_pool, 'final_avg_pool')
    reshaped_layer = tf.reshape(final_avg_pool, [-1, 64])
    final_dense = tf.layers.dense(
      inputs=reshaped_layer, units=self.num_classes)
    self.logits = tf.identity(final_dense, 'final_dense')

    self.loss_acc()

  def loss_acc(self):
    """The loss and accuracy of model."""
    with tf.name_scope("loss"):
      losses = tf.losses.sparse_softmax_cross_entropy(
        labels=self.labels, logits=self.logits)
      self.loss = tf.add(tf.reduce_mean(losses), 
        self.weight_decay * tf.add_n([tf.nn.l2_loss(v) 
          for v in tf.trainable_variables() if 'bias' not in v.name]))

    with tf.name_scope("accuracy"):
      correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels)
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def train_op(self):
    """The train operation."""
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      self.optimization = tf.train.AdamOptimizer(
        self.learning_rate).minimize(self.loss)


def main(unused_argv):

  detect_dir_is_existed(FLAGS.log_dir)
  # Read data files.
  train_data, train_labels, test_data, test_labels = cifar_read_data(
    "cifar-10-batches-py/", 5, 1)

  sess = tf.InteractiveSession()

  model = Resnet(
    image_size=FLAGS.image_size, img_depth=FLAGS.img_depth, 
    padded_size=FLAGS.padded_size, cropped_size=FLAGS.cropped_size, 
    num_classes=FLAGS.num_classes, weight_decay=FLAGS.weight_decay, 
    init_lr=FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, 
    decay_rate=FLAGS.decay_rate
    )

  tf.summary.scalar("Loss", model.loss)
  tf.summary.scalar('Accuracy', model.accuracy)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tests')
  saver = tf.train.Saver()

  tf.global_variables_initializer().run()
  saver = tf.train.Saver()

  for i in range(FLAGS.max_steps):
    if i % 100 == 0:
      summary, acc = sess.run(
        [merged, model.accuracy], 
        feed_dict={
                    model.inputs: test_data, 
                    model.labels: test_labels, 
                    model.phase: False
                  })
      test_writer.add_summary(summary, i)

      xs, ys = randomly_sample(train_data, train_labels, 
        batch_size=FLAGS.batch_size)
      summary, acc_t, _, _, l, lr = sess.run(
        [
          merged, model.accuracy, 
          model.optimization, 
          model.add_global, 
          model.loss, model.learning_rate
        ], 
        feed_dict={
                    model.inputs: xs, 
                    model.labels: ys, 
                    model.phase: True
                  })
      train_writer.add_summary(summary, i)

      format_str = (
        '%s: step %d, lr = %.7f, acc(train) = %.4f, acc(test) = %.4f')
      print(format_str % (datetime.now(), i, lr, acc_t, acc))
    else:
      xs, ys = randomly_sample(train_data, train_labels, 
        batch_size=FLAGS.batch_size)
      summary, _, _ = sess.run(
        [merged, model.optimization, model.add_global], 
        feed_dict={
                    model.inputs: xs, 
                    model.labels: ys, 
                    model.phase: True
                  })
      train_writer.add_summary(summary, i)


  # Save the model
  model_path = saver.save(sess, FLAGS.save_path)
  print("Model saved in file: %s" % model_path)

  train_writer.close()
  test_writer.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', type=bool, default=False, 
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=20000, 
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001, 
                      help='Initial learning rate.')
  parser.add_argument('--decay_steps', type=int, default=5000, 
                      help='The period of decay.')
  parser.add_argument('--decay_rate', type=float, default=0.65, 
                      help='The rate of decay.')
  parser.add_argument('--weight_decay', type=float, default=2e-6,
                      help='The rate of weight decay.')
  parser.add_argument('--batch_size', type=int, default=128, 
                      help='The size of batch.')
  parser.add_argument('--num_classes', type=int, default=10,
                      help='The number of classes.')
  parser.add_argument('--image_size', type=str, default=32,
                      help='The size of image.')
  parser.add_argument('--img_depth', type=int, default=3, 
                      help="The image depth.")
  parser.add_argument('--padded_size', type=int, default=40, 
                      help="The size of padded image.")
  parser.add_argument('--cropped_size', type=str, default=24, 
                      help="The size of cropped image.")
  parser.add_argument('--log_dir', type=str, 
                      default='logs/cifar_resnet_with_summaries', 
                      help='Summaries logs directory')
  parser.add_argument('--save_path', type=str,  
                      default='models/cifar_resnet.ckpt')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()