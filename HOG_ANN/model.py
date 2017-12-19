from __future__ import print_function

import tensorflow as tf


def weight_variable(shape, name):
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


class ANN(object):
    def __init__(self, num_classes, input_shape, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(
            tf.float32, [None, ] + input_shape, name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, ] + num_classes, name="input_y")
        l2_loss = tf.constant(0.0)

        with tf.name_scope('fc1'):
            W_fc1 = weight_variable(input_shape + [256], 'W_fc1')
            b_fc1 = bias_variable([256])
            h_fc1 = tf.nn.relu(tf.matmul(self.input_x, W_fc1) + b_fc1)

            l2_loss += tf.nn.l2_loss(W_fc1)
            l2_loss += tf.nn.l2_loss(b_fc1)
        with tf.name_scope('dropout1'):
            self.keep_prob_1 = tf.placeholder(tf.float32, name='keep_prob_1')
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob_1)
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
