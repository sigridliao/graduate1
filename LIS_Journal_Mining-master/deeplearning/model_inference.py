"""
The Inference Part of Model
"""
import tensorflow as tf


def get_input_node():
    return None


def get_output_node():
    return 15


def get_weight_variable(shape, regularizer=None):
    weights = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer)
    return weights


def get_bias_variable(shape):
    biases = tf.get_variable('bias', shape=shape, initializer=tf.constant_initializer(0.0))
    return biases


def inference(input_tensor, regularizer):
    """
    Build the Feed Forword Model and Hyperparameters
    """
    # Title: Convolutional Layer I
    with tf.variable_scope('title_conv1'):
        title_conv1_weights = get_weight_variable(None)
        title_conv1_biases = get_bias_variable(None)
        title_conv1 = tf.nn.conv2d(input=None, filter=None, strides=None, padding='SAME')
        title_conv1_relu = tf.nn.relu(tf.nn.bias_add(title_conv1, title_conv1_biases))

    # Title: Pooling Layer I
    with tf.variable_scope('title_pool1'):
        title_pool1 = tf.nn.max_pool(title_conv1_relu, ksize=None, strides=None, padding='SAME')

    # Title: Reshape Pooling Layer I
    pool_shape = title_pool1.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped_title = tf.reshape(title_pool1, [pool_shape[0], nodes])

    # Title: Full Connection Layer I
    with tf.variable_scope('title_fc1'):
        title_fc1_weights = get_weight_variable(None, None)
        title_fc1_biases = get_bias_variable(None)
        title_fc1 = tf.nn.relu(tf.matmul(reshaped_title, title_fc1_weights) + title_fc1_biases)
        title_fc1 = tf.nn.dropout(title_fc1, keep_prob=0.9)

    # Abstract: Convolutional Layer I
    with tf.variable_scope('abstract_conv1'):
        abstract_conv1_weights = get_weight_variable(None)
        abstract_conv1_biases = get_bias_variable(None)
        abstract_conv1 = tf.nn.conv2d(input=None, filter=None, strides=None, padding='SAME')
        abstract_conv1_relu = tf.nn.relu(tf.nn.bias_add(abstract_conv1, abstract_conv1_biases))

    # Abstract: Pooling Layer I
    with tf.variable_scope('abstract_pool1'):
        abstract_pool1 = tf.nn.max_pool(title_conv1_relu, ksize=None, strides=None, padding='SAME')

    # Abstract: Convolutional Layer II
    with tf.variable_scope('abstract_conv2'):
        abstract_conv_2_weights = get_weight_variable(None)
        abstract_conv_2_biases = get_bias_variable(None)
        abstract_conv_2 = tf.nn.conv2d(input=None, filter=None, strides=None, padding='SAME')
        abstract_conv_2_relu = tf.nn.relu(tf.nn.bias_add(abstract_conv1, abstract_conv1_biases))

    # Abstract: Pooling Layer II
    with tf.variable_scope('abstract_pool2'):
        abstract_pool2 = tf.nn.max_pool(title_conv1_relu, ksize=None, strides=None, padding='SAME')

    # Abstract: Reshape Pooling Layer II
    pool_shape = abstract_pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped_abstract = tf.reshape(abstract_pool2, [pool_shape[0], nodes])

    # Abstract: Full Connection Layer I
    with tf.variable_scope('abstract_fc1'):
        abstract_fc1_weights = get_weight_variable(None, None)
        abstract_fc1_biases = get_bias_variable(None)
        abstract_fc1 = tf.nn.relu(tf.matmul(reshaped_abstract, abstract_fc1_weights) + abstract_fc1_biases)

    # Keywords: Full Connection Layer I
    with tf.variable_scope('keywords_fc1'):
        reshaped_keywords = None
        keywords_fc1_weights = get_weight_variable(None, None)
        keywords_fc1_biases = get_bias_variable(None)
        keywords_fc1 = tf.nn.relu(tf.matmul(reshaped_keywords, keywords_fc1_weights) + keywords_fc1_biases)
        keywords_fc1 = tf.nn.dropout(keywords_fc1, keep_prob=0.9)

    # LSTM Model
    with tf.variable_scope('lstm'):
        lstm = tf.contrib.rnn.BasicLSTMCell(3)


    return abstract_fc1

