from __future__ import print_function

import argparse
import datetime
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import data_helpers
from CNN import model

FLAGS = None


def prepare_log_dir():
    '''Clears the log files then creates new directories to place
        the tensorbard log file.'''
    if tf.gfile.Exists(FLAGS.log_dir + '/logs'):
        tf.gfile.DeleteRecursively(FLAGS.log_dir + '/logs')
    tf.gfile.MakeDirs(FLAGS.log_dir + '/logs')


def train_model(Model, x_train, y_train, x_val, y_val, x_test, y_test, sess):
    global_step = tf.Variable(0, trainable=False)
    model = Model(num_classes=[10], input_shape=[
        32, 32, 3], l2_reg_lambda=1e-4)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(model.loss, global_step)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.log_dir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=FLAGS.num_checkpoints)
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.keep_prob_1: 0.5,
            model.keep_prob_2: 0.5,
            model.keep_prob_3: 0.5,
            model.keep_prob_4: 0.5,
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, model.loss, model.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(
            time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.keep_prob_1: 1.0,
            model.keep_prob_2: 1.0,
            model.keep_prob_3: 1.0,
            model.keep_prob_4: 1.0,
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, model.loss, model.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(
            time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    evaluate_every = (len(x_train) - 1) // FLAGS.batch_size + 1
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.max_steps)
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if x_val is not None and current_step % evaluate_every == 0:
            print("\nEvaluation:")
            dev_step(x_val, y_val, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix,
                              global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

    path = saver.save(sess, checkpoint_prefix,
                      tf.train.global_step(sess, global_step))
    print("Saved final model to {}\n".format(path))
    test_batches = data_helpers.batch_iter(
        list(x_test), FLAGS.batch_size, 1, shuffle=False)
    all_predictions = []
    for x_test_batch in test_batches:
        test_feed_dict = {
            model.keep_prob_1: 1.0,
            model.keep_prob_2: 1.0,
            model.keep_prob_3: 1.0,
            model.keep_prob_4: 1.0,
            model.input_x: x_test_batch
        }
        batch_predictions = sess.run(model.predictions, test_feed_dict)
        all_predictions = np.concatenate([all_predictions, batch_predictions])
    if y_test is not None:
        if y_test.ndim == 2:
            y_test = np.argmax(y_test, 1)
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(
            correct_predictions / float(len(y_test))))
        import pandas as pd
        from sklearn.metrics import classification_report, confusion_matrix
        con_mat = confusion_matrix(y_true=y_test, y_pred=all_predictions)
        class_rep = classification_report(
            y_true=y_test, y_pred=all_predictions)
        con_mat = pd.DataFrame(
            con_mat, index=np.arange(10), columns=np.arange(10))
        print('Classification report')
        print(class_rep)
        print('Confusion Matrix')
        print(con_mat)
        with open(FLAGS.log_dir + '/class_rep.txt', 'w') as file:
            file.write(class_rep)
        con_mat.to_csv(FLAGS.log_dir + '/con_mat.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='Checkpoint frequency')
    parser.add_argument('--num_checkpoints', type=int, default=20,
                        help='Number of checkpoints keep.')
    parser.add_argument('--validation_split', type=float,
                        default=0.1, help='Validation split.')
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    x_val, y_val = None, None
    x_train, y_train, x_test, y_test = data_helpers.load_svhn(FLAGS.data_dir)
    if 0 < FLAGS.validation_split < 1:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=FLAGS.validation_split, random_state=0)
    with tf.Session() as sess:
        train_model(model.ConvNet, x_train, y_train,
                    x_val, y_val, x_test, y_test, sess)
