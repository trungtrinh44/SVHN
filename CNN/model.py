from __future__ import print_function
import tensorflow as tf


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, kernel_width, kernel_height):
    return tf.nn.max_pool(x, ksize=[1, kernel_width, kernel_height, 1],
                          strides=[1, kernel_width, kernel_height, 1], padding='SAME')


def weight_variable(shape, name):
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


class ConvNet(object):
    def __init__(self, num_classes, input_shape, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(
            tf.float32, [None, ] + input_shape, name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, ] + num_classes, name="input_y")
        l2_loss = tf.constant(0.0)
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([3, 3, 3, 32], 'W_conv1')
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(self.input_x, W_conv1) + b_conv1)
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([3, 3, 32, 32], 'W_conv2')
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        with tf.name_scope('maxpool1'):
            h_pool1 = max_pool(h_conv2, 2, 2)
        with tf.name_scope('dropout1'):
            self.keep_prob_1 = tf.placeholder(tf.float32, name='keep_prob_1')
            h_pool1_drop = tf.nn.dropout(h_pool1, self.keep_prob_1)

        with tf.name_scope('conv3'):
            W_conv3 = weight_variable([5, 5, 32, 128], 'W_conv3')
            b_conv3 = bias_variable([128])
            h_conv3 = tf.nn.relu(conv2d(h_pool1_drop, W_conv3) + b_conv3)
        with tf.name_scope('conv4'):
            W_conv4 = weight_variable([5, 5, 128, 128], 'W_conv4')
            b_conv4 = bias_variable([128])
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
        with tf.name_scope('maxpool2'):
            h_pool2 = max_pool(h_conv4, 2, 2)
        with tf.name_scope('dropout2'):
            self.keep_prob_2 = tf.placeholder(tf.float32, name='keep_prob_2')
            h_pool2_drop = tf.nn.dropout(h_pool2, self.keep_prob_2)

        with tf.name_scope('conv5'):
            W_conv5 = weight_variable([7, 7, 128, 256], 'W_conv5')
            b_conv5 = bias_variable([256])
            h_conv5 = tf.nn.relu(conv2d(h_pool2_drop, W_conv5) + b_conv5)
        with tf.name_scope('maxpool3'):
            h_pool3 = max_pool(h_conv5, 4, 4)
        with tf.name_scope('dropout3'):
            self.keep_prob_3 = tf.placeholder(tf.float32, name='keep_prob_3')
            h_pool3_drop = tf.nn.dropout(h_pool3, self.keep_prob_3)

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([1024, 256], 'W_fc1')
            b_fc1 = bias_variable([256])

            h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 1024])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

            l2_loss += tf.nn.l2_loss(W_fc1)
            l2_loss += tf.nn.l2_loss(b_fc1)
        with tf.name_scope('dropout4'):
            self.keep_prob_4 = tf.placeholder(tf.float32, name='keep_prob_4')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob_4)
        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([256, ] + num_classes, 'W_fc2')
            b_fc2 = bias_variable(num_classes)
            l2_loss += tf.nn.l2_loss(W_fc2)
            l2_loss += tf.nn.l2_loss(b_fc2)
            self.scores = tf.add(
                tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                    logits=self.scores)
            self.loss = tf.add(tf.reduce_mean(cross_entropy),
                               l2_reg_lambda * l2_loss, name='loss')

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
