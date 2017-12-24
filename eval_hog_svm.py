import argparse

import numpy as np
import pandas as pd
from skimage import color
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

import data_helpers


def apply_hog(image):
    image = color.rgb2gray(image)
    hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), transform_sqrt=True,
                    block_norm='L2')
    return hog_image


def hog_images(images):
    return np.vstack([apply_hog(x) for x in images])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory for storing input data')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./HOG_SVM/final_model/svm.pkl',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        default='./HOG_SVM',
        type=str,
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    _, _, x_test, y_test = data_helpers.load_shvn_for_svm(FLAGS.data_dir)
    x_hog_test = hog_images(x_test)
    sgdc = joblib.load('./HOG_SVM/final_model/svm.pkl')
    y_preds = sgdc.predict(x_hog_test)
    print(accuracy_score(y_preds, y_test))
    con_mat = confusion_matrix(y_true=y_test, y_pred=y_preds)
    class_rep = classification_report(y_true=y_test, y_pred=y_preds)
    con_mat = pd.DataFrame(con_mat, index=np.arange(10), columns=np.arange(10))
    print('Classification report')
    print(class_rep)
    print('Confusion Matrix')
    print(con_mat)
    with open(FLAGS.log_dir + '/class_rep.txt', 'w') as file:
        file.write(class_rep)
    con_mat.to_csv(FLAGS.log_dir + '/con_mat.csv')
