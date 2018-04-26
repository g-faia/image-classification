import sys
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('..')

import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from helper import *


class Logist(object):

  def __init__(self, 
    input_size, num_classes, init_lr, 
    decay_steps, decay_rate, weight_decay):
    
    self.weight_decay = weight_decay
    self.inputs = tf.placeholder(tf.float32, [None, input_size], name='inputs')
    self.labels = tf.placeholder(tf.int64, [None], name='labels')
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(
      init_lr, global_step=self.global_step, 
      decay_steps=decay_steps, decay_rate=decay_rate)

    with tf.name_scope('inference'):
      weights = tf.Variable(tf.truncated_normal(
        shape=[input_size, num_classes], mean=0, stddev=0.1, name='W'))
      biases = tf.Variable(tf.constant(
        shape=[num_classes], value=0.1, name='bias'))
      self.logits = tf.matmul(self.inputs, weights) + biases

    self.loss_acc()
    self.train_op()

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
  
  model = Logist(
    input_size=FLAGS.input_size, num_classes=FLAGS.num_classes, 
    init_lr=FLAGS.learning_rate, decay_steps=FLAGS.decay_steps, 
    decay_rate=FLAGS.decay_rate, weight_decay=FLAGS.weight_decay
    )

  tf.summary.scalar("Loss", model.loss)
  tf.summary.scalar('Accuracy', model.accuracy)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/tests')

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', type=bool, default=False, 
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=5000, 
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
  parser.add_argument('--input_size', type=int, default=784,
                      help='The size of input.')
  parser.add_argument('--num_classes', type=int, default=10,
                      help='The number of classes.')
  parser.add_argument('--image_size', type=str, default=28,
                      help='The size of image.')
  parser.add_argument('--log_dir', type=str, 
                      default='logs/mnist_logist_with_summaries', 
                      help='Summaries logs directory')
  parser.add_argument('--save_path', type=str,  
                      default='models/mnist_logist.ckpt')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()
