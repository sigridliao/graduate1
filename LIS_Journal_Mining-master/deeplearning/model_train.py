"""
The Training Part of Model
"""

import tensorflow as tf
import os
from deeplearning.model_inference import get_input_node
from deeplearning.model_inference import get_output_node
from deeplearning.model_inference import inference

os.chdir('E:\HDQ\Projects\Graduation')

# Parameters
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# Save Model
MODEL_SAVE_PATH = './nn_model/'
MODEL_NAME = 'SCLSTMSA.ckpt'


def get_moving_average_decay():
    return MOVING_AVERAGE_DECAY


def get_model_save_path():
    return MODEL_SAVE_PATH


def train(data, labels):
    """
    Fit the data with the model
    """
    x = tf.placeholder(tf.float32, [None, get_input_node()])
    y_ = tf.placeholder(tf.float32, [None, get_output_node()])

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, None, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINING_STEPS):
            xs, ys = data.next_batch(BATCH_SIZE)  # need to be configure
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('Training Step %d, loss on training batch %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    """
    Load the data and train the model
    """
    data = list()
    labels = list()


if __name__ == '__main__':
    tf.app.run()

