"""
The Evaluation Part of Model
"""

import tensorflow as tf
import time
from deeplearning.model_inference import get_input_node
from deeplearning.model_inference import get_output_node
from deeplearning.model_inference import inference
from deeplearning.model_train import get_moving_average_decay
from deeplearning.model_train import get_model_save_path

EVAL_INTERVAL_SECS = 10


def evaluate(data):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, get_input_node()], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, get_output_node()], name='y-input')
        validate_feed = {x: data.validation_data, y_: data.validation_label}

        y = inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(get_moving_average_decay())
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(get_model_save_path())
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('Training Step %s, accuracy is %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main():
    data = list()
    labels = list()

if __name__ == '__main__':
    tf.app.run()