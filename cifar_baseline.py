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


class CNNBaseline(object):

  def __init__(self, 
    image_size, img_depth, cropped_size, num_classes, init_lr, 
    weight_decay, decay_steps, decay_rate):

    self.num_classes = num_classes
    self.weight_decay = weight_decay
    
    self.inputs = tf.placeholder(
      tf.float32, [None, image_size, image_size, img_depth], name='inputs')
    self.phase = tf.placeholder(dtype=tf.bool, shape=(), name='phase')
    with tf.device('/cpu:0'):
      self.distorted_images = pre_process_images(
        images=self.inputs, phase=self.phase, img_size_cropped=cropped_size)   
    
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
    """convolution module."""
    with tf.name_scope('conv1'):
      conv1_filters = weight_variable(
        shape=[5, 5, 3, 32], mean=0, stddev=1e-2, name='filters')    
      conv1_biases = bias_variable(shape=[32], constant=0.1, name='bias')
      conv1_pre = tf.nn.conv2d(
        input=self.distorted_images, filter=conv1_filters, 
        strides=[1, 1, 1, 1], padding='SAME')
      # conv1_pre = tf.nn.conv2d(
        # input=self.inputs, filter=conv1_filters, strides=[1, 1, 1, 1], 
        # padding='SAME')
      conv1_bn = tf.layers.batch_normalization(
        inputs=conv1_pre + conv1_biases, training=self.phase, name='conv1_bn')
      conv1_outputs = tf.nn.relu(conv1_bn)
      # conv1_outputs = tf.nn.relu(conv1_pre)

    with tf.name_scope('pool1'):
      pool1 = tf.nn.max_pool(
        conv1_outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
        padding='SAME', name='pool1')

    with tf.name_scope('conv2'):
      conv2_filters = weight_variable(
        shape=[5, 5, 32, 64], mean=0, stddev=1e-2, name='filters')    
      conv2_biases = bias_variable(shape=[64], constant=0.1, name='bias')
      conv2_pre = tf.nn.conv2d(
        input=pool1, filter=conv2_filters, strides=[1, 1, 1, 1], 
        padding='SAME')
      conv2_bn = tf.layers.batch_normalization(
        inputs=conv2_pre + conv2_biases, training=self.phase, name='conv2_bn')
      conv2_outputs = tf.nn.relu(conv2_bn)
      # conv2_outputs = tf.nn.relu(conv2_pre)

    with tf.name_scope('pool2'):
      pool2 = tf.nn.max_pool(
        conv2_outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
        padding='SAME', name='pool2')

    with tf.name_scope('fc'):
      fc = tf.reshape(pool2, shape=[-1, 6*6*64])
      # fc = tf.reshape(pool2, shape=[-1, 8*8*64])
    
    fc1_layer = nn_layer(
      input_tensor=fc, input_dim=6*6*64, output_dim=1000, 
      layer_name='fc1_layer')
    # fc1_layer = nn_layer(
      # input_tensor=fc, input_dim=8*8*64, output_dim=1000, 
      # layer_name='fc1_layer')
    self.logits = nn_layer(
      input_tensor=fc1_layer, input_dim=1000, output_dim=self.num_classes,
      layer_name='fc2_layer', act=tf.identity)

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

  model = CNNBaseline(
    num_classes=FLAGS.num_classes, image_size=FLAGS.image_size, 
    img_depth=FLAGS.img_depth, cropped_size=FLAGS.cropped_size, 
    init_lr=FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, 
    decay_rate=FLAGS.decay_rate, weight_decay=FLAGS.weight_decay
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
  parser.add_argument(
    '--fake_data', nargs='?', const=True, type=bool, default=False, 
    help='If true, uses fake data for unit testing.'
    )
  parser.add_argument(
    '--max_steps', type=int, default=20000, 
    help='Number of steps to run trainer.'
    )
  parser.add_argument(
    '--learning_rate', type=float, default=0.001, 
    help='Initial learning rate'
    )
  parser.add_argument(
    '--decay_steps', type=int, default=5000, 
    help='the period of decay'
    )
  parser.add_argument(
    '--decay_rate', type=float, default=0.65, 
    help='the rate of decay'
    )
  parser.add_argument(
    '--weight_decay', type=float, default=2e-6
    )
  parser.add_argument(
    '--batch_size', type=int, default=128, 
    help='the size of batch'
    )
  parser.add_argument(
    '--num_classes', type=int, default=10
    )
  parser.add_argument(
    '--image_size', type=int, default=32
    )
  parser.add_argument(
    '--img_depth', type=int, default=3
    )
  parser.add_argument(
    '--cropped_size', type=str, default=24
    )
  parser.add_argument(
    '--log_dir', type=str, default='logs/cifar_baseline_with_summaries', 
    help='Summaries logs directory'
    )
  parser.add_argument(
    '--save_path', type=str,  default='models/cifar_basline.ckpt'
    )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()