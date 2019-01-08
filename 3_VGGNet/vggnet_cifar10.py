"""VGGNet - 90.5%"""
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


class VGGNet(object):

  def __init__(self, image_size, img_depth, num_classes, 
               dropout, init_lr, weight_decay):

    self.num_classes = num_classes
    self.dropout = dropout
    
    self.inputs = tf.placeholder(tf.float32, [None, image_size, image_size, img_depth])

    self.mode = tf.placeholder(dtype=tf.bool, shape=(), name='mode')

    with tf.device('/cpu:0'):
      # use the distorted images as the input of model.
      self.distorted_images = helper.pre_process_images(images=self.inputs, 
                                                        phase=self.mode,
                                                        image_size=image_size)

    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    
    self.regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    self.global_step = tf.Variable(0, trainable=False)
    boundaries = [10000, 20000, 30000]
    values = [init_lr / (10 ** i) for i in range(len(boundaries) + 1)]
    self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
    self.add_global = self.global_step.assign_add(1)

    self.model(), self.loss(), self.train_op()

  def model(self):
    """(conv3x3 -> conv3x3 -> pool2x2) blocks -> multiple fc."""
    def _conv(inputs, n_filters):
      conv = tf.layers.batch_normalization(inputs, training=self.mode)
      conv = tf.layers.conv2d(conv, n_filters, (3, 3), 
                              activation=tf.nn.relu,
                              padding='SAME',
                              kernel_regularizer=self.regularizer)
      return conv

    def _vgg_block(inputs, n_filters):
      conv = _conv(inputs, n_filters)
      conv = _conv(inputs, n_filters)
      pool = tf.layers.max_pooling2d(conv, 2, 2)
      return pool

    block1 = _vgg_block(self.distorted_images, n_filters=64)
    block2 = _vgg_block(block1, n_filters=128)
    block3 = _vgg_block(block2, n_filters=128)

    flatten = tf.layers.flatten(block3)

    fc1 = tf.layers.batch_normalization(flatten, training=self.mode)
    fc1 = tf.layers.dense(fc1, 1024,
                          activation=tf.nn.relu, 
                          kernel_regularizer=self.regularizer)
    fc1 = tf.layers.dropout(fc1, rate=self.dropout, training=self.mode)

    self.logits = tf.layers.dense(fc1, self.num_classes,
                                  kernel_regularizer=self.regularizer)
    
  def loss(self):
    """The loss and accuracy of model."""
    with tf.name_scope("loss"):
      losses = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
      self.loss = tf.reduce_mean(losses) + tf.losses.get_regularization_loss()

  def train_op(self):
    """The train operation."""
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      self.optimization = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def main(unused_argv):

  # Load cifar-10 dataset.
  train_data, train_labels, test_data, test_labels = helper.cifar_data_loader()

  model = VGGNet(
    num_classes=FLAGS.num_classes, image_size=FLAGS.image_size, img_depth=FLAGS.img_depth, 
    dropout=FLAGS.dropout, init_lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay
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
  parser.add_argument('--learning_rate', type=float, default=0.001, 
                      help='Initial learning rate.')
  parser.add_argument('--weight_decay', type=float, default=2e-4,
                      help='The rate of weight decay.')
  parser.add_argument('--batch_size', type=int, default=128, 
                      help='The size of batch.')
  parser.add_argument('--num_classes', type=int, default=10,
                      help='The number of classes.')
  parser.add_argument('--image_size', type=str, default=32,
                      help='The size of image.')
  parser.add_argument('--img_depth', type=int, default=3, 
                      help="The image depth.")
  parser.add_argument('--dropout', type=float, default=0.5, 
                      help='Keep probability for training dropout.')
  parser.add_argument('--save_path', type=str,  
                      default='models/alexnet_cifar.ckpt')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()
