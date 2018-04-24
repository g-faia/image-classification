"""The final mnist model mnist_baseline_v2 2018-1-22
Becasue the number of layers of model is too small, so the batch-normalization is not usefull.
"""
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
    input_size, image_size, num_classes, init_lr, 
    weight_decay, decay_steps, decay_rate):

    self.input_size = input_size
    self.num_classes = num_classes
    self.weight_decay = weight_decay
    self.inputs = tf.placeholder(
      tf.float32, [None, self.input_size], name='inputs')
    self.reshaped_inputs = tf.reshape(
      self.inputs, [-1, image_size, image_size, 1], name="reshaped_inputs")    
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(
      init_lr, global_step=self.global_step, 
      decay_steps=decay_steps, decay_rate=decay_rate)
    # self.is_training = tf.placeholder(tf.bool)

    self.model()
    self.train_op()

  def model(self):
    """Build the  mnist model."""
    # the convolutional layers
    with tf.name_scope('conv1'):
      conv1_filters = weight_variable(shape=[5, 5, 1, 32], stddev=5e-2)
      conv1_biases = bias_variable(shape=[32], constant=0.0)
      conv1_pre = tf.nn.conv2d(
        input=self.reshaped_inputs, filter=conv1_filters, strides=[1, 1, 1, 1],
        padding='VALID')
      # conv1_bn = tf.layers.batch_normalization(
        # inputs=conv1_pre + conv1_biases, axis=-1, training=self.is_training)
      # conv1_outputs = tf.nn.relu(conv1_bn)
      conv1_outputs = tf.nn.relu(conv1_pre)

    with tf.name_scope('pool1'):
      pool1_outputs = tf.nn.max_pool(
        value=conv1_outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
        padding='SAME')
  
    with tf.name_scope('conv2'):
      conv2_filters = weight_variable(shape=[5, 5, 32, 64], stddev=5e-2)
      conv2_biases = bias_variable(shape=[64], constant=0.0)
      conv2_pre = tf.nn.conv2d(
        input=pool1_outputs, filter=conv2_filters, strides=[1, 1, 1, 1], 
        padding='VALID')
      # conv2_bn = tf.layers.batch_normalization(
        # inputs=conv2_pre + conv2_biases, axis=-1, training=self.is_training)
      # conv2_outputs = tf.nn.relu(conv2_bn)
      conv2_outputs = tf.nn.relu(conv2_pre)

    with tf.name_scope('pool2'):
      pool2_outputs = tf.nn.max_pool(
        value=conv2_outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
        padding='SAME')

    # the multi-perceptron
    with tf.name_scope('full_connected'):
      all_connected = tf.reshape(pool2_outputs, shape=[-1, 64 * 4 * 4])
      with tf.name_scope('fc1'):
        fc1_weights = weight_variable(shape=[64 * 4 * 4, 1000], stddev=0.04)
        fc1_baises = bias_variable(shape=[1000], constant=.0)
        fc1_pre = tf.matmul(all_connected, fc1_weights) + fc1_baises
        fc1_outputs = tf.nn.relu(fc1_pre)
    
      with tf.name_scope('fc2'):
        fc2_weights = weight_variable(shape=[1000, 100], stddev=0.04)
        fc2_baises = bias_variable(shape=[100], constant=.0)
        fc2_pre = tf.matmul(fc1_outputs, fc2_weights) + fc2_baises
        fc2_outputs = tf.nn.relu(fc2_pre)
    
      with tf.name_scope('fc3'):
        fc3_weights = weight_variable(
          shape=[100, self.num_classes], stddev=1 / 100.0)
        fc3_baises = bias_variable(shape=[self.num_classes], constant=.0)
        fc3_pre = tf.matmul(fc2_outputs, fc3_weights) + fc3_baises
        self.logits = tf.identity(fc3_pre)

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
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  sess = tf.InteractiveSession()

  model = CNNBaseline(
    input_size=FLAGS.input_size, num_classes=FLAGS.num_classes, 
    image_size=FLAGS.image_size, init_lr=FLAGS.learning_rate, 
    decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, 
    weight_decay=FLAGS.weight_decay
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

      summary, acc, l, lr = sess.run(
        [merged, model.accuracy, model.loss, model.learning_rate], 
        feed_dict={
                    model.inputs: eval_data, 
                    model.labels: eval_labels
                  })
      test_writer.add_summary(summary, i)
      print('- Step {0}, - learning_rate : {1},'
            '- Accuracy : {2}, - Loss : {3}' 
            .format(i, lr, acc, l))

      xs, ys = mnist.train.next_batch(
        FLAGS.batch_size, fake_data=FLAGS.fake_data)
      summary, _, _ = sess.run(
        [merged, model.optimization, model.add_global], 
        feed_dict={
                    model.inputs: xs, 
                    model.labels: ys
                  })

      train_writer.add_summary(summary, i)
    else:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      summary, _, _ = sess.run(
        [merged, model.optimization, model.add_global], 
        feed_dict={
                    model.inputs: xs, 
                    model.labels: ys
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
    '--max_steps', type=int, default=5000, 
    help='Number of steps to run trainer.'
    )
  parser.add_argument(
    '--learning_rate', type=float, default=0.001, 
    help='Initial learning rate.'
    )
  parser.add_argument(
    '--decay_steps', type=int, default=5000, 
    help='The period of decay.'
    )
  parser.add_argument(
    '--decay_rate', type=float, default=0.65, 
    help='The rate of decay.'
    )
  parser.add_argument(
    '--weight_decay', type=float, default=2e-6,
    help='The rate of weight decay.'
    )
  parser.add_argument(
    '--batch_size', type=int, default=128, 
    help='The size of batch.'
    )
  parser.add_argument(
    '--input_size', type=int, default=784,
    help='The size of input.'
    )
  parser.add_argument(
    '--num_classes', type=int, default=10,
    help='The number of classes.'
    )
  parser.add_argument(
    '--image_size', type=str, default=28,
    help='The size of image.'
    )
  parser.add_argument(
    '--log_dir', type=str, default='logs/mnist_baseline_with_summaries', 
    help='Summaries logs directory'
    )
  parser.add_argument(
    '--save_path', type=str,  default='models/mnist_basline.ckpt'
    )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()
