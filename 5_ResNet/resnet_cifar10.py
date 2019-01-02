from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('..')

import os
import time
import argparse
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import helper


class Resnet(object):

  def __init__(self, image_size, img_depth, num_classes, 
               dropout, weight_decay, init_lr):

    self.num_classes = num_classes
    self.weight_decay = weight_decay
    self.dropout = dropout

    self.num_blocks = (image_size - 2) // 6
    self.inputs = tf.placeholder(tf.float32, [None, image_size, image_size, img_depth], 
                                 name='inputs')
    
    self.mode = tf.placeholder(dtype=tf.bool, shape=(), name='mode')

    with tf.device('/cpu:0'):
      self.distorted_images = helper.pre_process_images(images=self.inputs, 
                                                        phase=self.mode, 
                                                        image_size=image_size)
    
    # use the distorted images as the input of model.
    self.labels = tf.placeholder(tf.int64, [None], name='labels')

    self.regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    self.global_step = tf.Variable(0, trainable=False)
    boundaries = [40000, 60000, 80000]
    values = [init_lr / (10 ** i) for i in range(len(boundaries) + 1)]
    self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
    self.add_global = self.global_step.assign_add(1)

    self.model(), self.loss(), self.train_op()

  def model(self):

    def _conv(inputs, n_filters, strides):
      initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0/(9*n_filters)))
      conv = tf.layers.conv2d(inputs, n_filters, (3, 3), strides, 
                              padding='SAME', use_bias=False,
                              kernel_regularizer=self.regularizer,
                              kernel_initializer=initializer)
      return conv

    def _res_subblock(inputs, n_filters, strides, activate_before_res=False):
      if activate_before_res:
        inputs = tf.layers.batch_normalization(inputs, training=self.mode)
        inputs = tf.nn.relu(inputs)
        shortcut = inputs
      else:
        shortcut = inputs
        inputs = tf.layers.batch_normalization(inputs, training=self.mode)
        inputs = tf.nn.relu(inputs)

      conv = _conv(inputs, n_filters, strides)

      conv = tf.layers.batch_normalization(conv, training=self.mode)
      conv = tf.nn.relu(conv)
      conv = _conv(conv, n_filters, 1)

      in_channels = int(shortcut.shape[-1])
      if in_channels != n_filters:
        shortcut = tf.layers.average_pooling2d(shortcut, strides, strides, padding='VALID')
        shortcut = tf.pad(shortcut, 
                          [[0, 0], [0, 0], [0, 0],
                           [(n_filters-in_channels)//2, (n_filters-in_channels)//2]])
      
      conv += shortcut

      return conv

    def _res_block(inputs, n_filters, strides, activate_before_res):
      conv = _res_subblock(inputs, n_filters, strides, activate_before_res)
      for i in range(1, self.num_blocks):
        conv = _res_subblock(conv, n_filters, 1)
      return conv

    init_conv = _conv(self.distorted_images, 16, 1)
    block1 = _res_block(init_conv, 16, 1, True)
    block2 = _res_block(block1, 32, 2, False)
    block3 = _res_block(block2, 64, 2, False)

    final_norm = tf.layers.batch_normalization(block3, training=self.mode)
    final_norm = tf.nn.relu(final_norm)
    global_avg = tf.layers.average_pooling2d(final_norm, 8, 1, padding='VALID')

    flatten = tf.layers.flatten(global_avg)
    self.logits = tf.layers.dense(flatten, self.num_classes,
                                  kernel_regularizer=self.regularizer)

  def loss(self):
    """The loss and accuracy of model."""
    with tf.name_scope("loss"):
      losses = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
      self.loss = tf.reduce_mean(losses)

  def train_op(self):
    """The train operation."""
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      # self.optimization = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
      self.optimization = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss)


def main(unused_argv):

  # Read data files.
  train_data, train_labels, test_data, test_labels = helper.cifar_data_loader()

  model = Resnet(
    image_size=FLAGS.image_size, img_depth=FLAGS.img_depth, 
    num_classes=FLAGS.num_classes, dropout=FLAGS.dropout,
    weight_decay=FLAGS.weight_decay, init_lr=FLAGS.learning_rate, 
    )

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  for e in range(FLAGS.epochs):
    print("----- Epoch {}/{} -----".format(e + 1, FLAGS.epochs))
    # training stage. 
    train_batches = helper.generate_batches(train_data, train_labels, FLAGS.batch_size)
    for xt, yt in tqdm(train_batches, desc="Training", ascii=True):
      _, i = sess.run([model.optimization, model.add_global], 
                      feed_dict={ model.inputs: xt, model.labels: yt, model.mode: True})
    
    # testing stage, use mini-batch mode to inference all test data.
    test_batches = helper.generate_batches(test_data, test_labels, FLAGS.batch_size)
    total_pred = correct_pred = 0
    total_loss = []
    for xt, yt in test_batches:
      logits, loss, lr = sess.run([model.logits, model.loss, model.learning_rate], 
                                  feed_dict={ model.inputs: xt, model.labels: yt, 
                                              model.mode: False})

      pred = np.argmax(logits, axis=1)
      correct_pred += np.sum(yt == pred)
      total_pred += yt.shape[0]
      total_loss.append(loss)

    acc = correct_pred / total_pred
    loss = np.mean(total_loss)

    current = time.asctime(time.localtime(time.time()))
    print("""{0} Step {1:5} Learning rate: {2:.6f} Losss: {3:.4f} Accuracy: {4:.4f}"""
          .format(current, i, lr, loss, acc))

  # Save the model
  saver = tf.train.Saver()
  model_path = saver.save(sess, FLAGS.save_path)
  print("Model saved in file: %s" % model_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=500,
                      help='Number of epochs to run trainer.')  
  parser.add_argument('--learning_rate', type=float, default=0.1, 
                      help='Initial learning rate.')
  parser.add_argument('--weight_decay', type=float, default=2e-4,
                      help='The rate of weight decay.')
  parser.add_argument('--batch_size', type=int, default=128, 
                      help='The size of batch.')
  parser.add_argument('--dropout', type=float, default=0.5, 
                      help='Keep probability for training dropout.')
  parser.add_argument('--num_classes', type=int, default=10,
                      help='The number of classes.')
  parser.add_argument('--image_size', type=str, default=32,
                      help='The size of image.')
  parser.add_argument('--img_depth', type=int, default=3, 
                      help="The image depth.")
  parser.add_argument('--save_path', type=str,  
                      default='models/cifar_resnet.ckpt')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()
