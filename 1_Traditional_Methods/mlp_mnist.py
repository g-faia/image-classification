"""Multilayer perceptron - Accuracy: 98.2%"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('..')

import os
import time
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import helper


class MLP(object):

  def __init__(self, input_size, num_classes, 
               dropout, init_lr, decay_steps, decay_rate, weight_decay):
    
    self.input_size = input_size
    self.num_classes = num_classes
    self.dropout = dropout

    self.inputs = tf.placeholder(tf.float32, [None, self.input_size], name='inputs')
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    
    self.mode = tf.placeholder(dtype=tf.bool, shape=(), name='mode')
    self.regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(init_lr, global_step=self.global_step, 
                                                    decay_steps=decay_steps, decay_rate=decay_rate)

    self.model(), self.loss(), self.train_op()

  def model(self):
    """multiple fully connected."""
    fc1 = tf.layers.dense(self.inputs, 392, 
                          activation=tf.nn.relu, 
                          kernel_regularizer=self.regularizer)
    fc1 = tf.layers.dropout(fc1, rate=self.dropout, training=self.mode)
    
    fc2 = tf.layers.dense(fc1, 196, 
                          activation=tf.nn.relu, 
                          kernel_regularizer=self.regularizer)
    fc2 = tf.layers.dropout(fc2, rate=self.dropout, training=self.mode)
    
    fc3 = tf.layers.dense(fc2, 98, 
                          activation=tf.nn.relu, 
                          kernel_regularizer=self.regularizer)
    fc3 = tf.layers.dropout(fc3, rate=self.dropout, training=self.mode)

    self.logits = tf.layers.dense(fc3, self.num_classes, 
                                  kernel_regularizer=self.regularizer)

  def loss(self):
    """The loss and accuracy of model."""
    with tf.name_scope("loss"):
      losses = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
      self.loss = tf.reduce_mean(losses) + tf.losses.get_regularization_loss()

  def train_op(self):
    """The train operation."""
    self.optimization = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def main(unused_argv):
  
  train_data, train_labels, test_data, test_labels = helper.mnist_data_loader()  

  model = MLP(
    input_size=FLAGS.input_size,  num_classes=FLAGS.num_classes,
    dropout = FLAGS.dropout, init_lr=FLAGS.learning_rate, 
    decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, weight_decay=FLAGS.weight_decay
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
    
    # testing stage.
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001, 
                      help='Initial learning rate.')
  parser.add_argument('--decay_steps', type=int, default=5000, 
                      help='The period of decay.')
  parser.add_argument('--decay_rate', type=float, default=0.65, 
                      help='The rate of decay.')
  parser.add_argument('--weight_decay', type=float, default=2e-4,
                      help='The rate of weight decay.')
  parser.add_argument('--batch_size', type=int, default=128, 
                      help='The size of batch.')
  parser.add_argument('--input_size', type=int, default=784,
                      help='The size of input.')
  parser.add_argument('--num_classes', type=int, default=10,
                      help='The number of classes.')
  parser.add_argument('--dropout', type=float, default=0.5, 
                      help='Keep probability for training dropout.')
  parser.add_argument('--save_path', type=str,  
                      default='models/mnist_multi.ckpt')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()
