import tensorflow as tf
import numpy as np
from data_helpers import load_svhn, batch_iter
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import argparse
FLAGS = None


def main():
    x_train, y_train, x_test, y_test = load_svhn(FLAGS.data_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph(
                '{}.meta'.format(FLAGS.model_dir))
            saver.restore(sess, FLAGS.model_dir)
            keep_prob_1 = graph.get_operation_by_name(
                'dropout1/keep_prob_1').outputs[0]
            keep_prob_2 = graph.get_operation_by_name(
                'dropout2/keep_prob_2').outputs[0]
            keep_prob_3 = graph.get_operation_by_name(
                'dropout3/keep_prob_3').outputs[0]
            keep_prob_4 = graph.get_operation_by_name(
                'dropout4/keep_prob_4').outputs[0]
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            predictions = graph.get_operation_by_name(
                'fc2/predictions').outputs[0]
            test_batches = batch_iter(list(x_test), 64, 1, shuffle=False)
            all_predictions = []
            for x_test_batch in test_batches:
                test_feed_dict = {
                    keep_prob_1: 1.0,
                    keep_prob_2: 1.0,
                    keep_prob_3: 1.0,
                    keep_prob_4: 1.0,
                    input_x: x_test_batch
                }
                batch_predictions = sess.run(predictions, test_feed_dict)
                all_predictions = np.concatenate(
                    [all_predictions, batch_predictions])
            if y_test.ndim == 2:
                y_test = np.argmax(y_test, 1)
            correct_predictions = float(sum(all_predictions == y_test))
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(
                correct_predictions / float(len(y_test))))
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
    parser.add_argument(
        '--data_dir',
        default='./data',
        type=str,
        help='Directory for storing input data')
    parser.add_argument(
        '--model_dir',
        default='./CNN/final_model/final-model',
        type=str,
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        default='./',
        type=str,
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    main()
