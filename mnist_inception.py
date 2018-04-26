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


class Inception(object):

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

    self.model()
    self.train_op()

  def model(self):
    """Build the  mnist model.
    First, before using the inception module, extract the feature maps
    From the reshaped inputs. 
    """
    # Extract the feature maps.
    conv1 = conv2d_layer(
      inputs=self.reshaped_inputs, in_channels=1, out_channels=32, 
      filters_size=[5, 5], layer_name='conv1', padding='VALID')
    pool1 = max_pool_layer(inputs=conv1, layer_name='pool1')
    conv2 = conv2d_layer(
      inputs=pool1, in_channels=32, out_channels=64, 
      filters_size=[3, 3], layer_name='conv2', padding='VALID')
    pool2 = max_pool_layer(inputs=conv2, layer_name='pool2')

    # Because of huge memory usage, reduce the complexity of inception model.
    # Simpler inception module consist of four independent inner network(IN).
    # First IN, conv1x1
    # Second IN, conv1x1 -> conv2x2
    # Third IN, conv1x1 -> conv3x3
    # Forth IN, pool3x3 -> conv1x1
    with tf.name_scope('inc1'):
      inc11_conv1 = conv2d_layer(
        inputs=pool2, in_channels=64, out_channels=32, 
        filters_size=[1, 1], layer_name='inc11_conv1')
      
      inc12_conv1 = conv2d_layer(
        inputs=pool2, in_channels=64, out_channels=16, 
        filters_size=[1, 1], layer_name='inc12_conv1')
      inc12_conv2 = conv2d_layer(
        inputs=inc12_conv1, in_channels=16, out_channels=32, 
        filters_size=[2, 2], layer_name='inc12_conv2')
    
      inc13_conv1 = conv2d_layer(
        inputs=pool2, in_channels=64, out_channels=16, 
        filters_size=[1, 1], layer_name='inc13_conv1')
      inc13_conv2 = conv2d_layer(
        inputs=inc13_conv1, in_channels=16, out_channels=32, 
        filters_size=[3, 3], layer_name='inc13_conv2')
    
      inc14_pool1 = max_pool_layer(
        inputs=pool2, layer_name='inc14_conv1', 
        k_size=[1, 3, 3, 1], strides=[1, 1, 1, 1])
      inc14_conv1 = conv2d_layer(
        inputs=inc14_pool1, in_channels=64, out_channels=32, 
        filters_size=[1, 1], layer_name='inc14_conv1')

      inc1_o = [inc11_conv1, inc12_conv2, inc13_conv2, inc14_conv1]
      inc1_con = tf.concat(inc1_o, 3)

    # Full connected layer.
    with tf.name_scope('fc'):
      fc = tf.reshape(inc1_con, shape=[-1, 5*5*128])
      fc1_layer = nn_layer(
        input_tensor=fc, input_dim=5*5*128, output_dim=1000, 
        layer_name='fc1_layer')
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

  model = Inception(
    input_size=FLAGS.input_size, num_classes=FLAGS.num_classes, 
    image_size=FLAGS.image_size, init_lr=FLAGS.learning_rate, 
    decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, 
    weight_decay=FLAGS.weight_decay)

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
            ' - Accuracy : {2}, - Loss : {3}' 
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
                      default='logs/mnist_inception_with_summaries', 
                      help='Summaries logs directory')
  parser.add_argument('--save_path', type=str,  
                      default='models/mnist_inception.ckpt')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run()
