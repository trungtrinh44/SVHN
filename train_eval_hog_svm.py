import argparse

import numpy as np
from skimage import color
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

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
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    x_train, y_train, x_test, y_test = data_helpers.load_shvn_for_svm(FLAGS.data_dir)
    x_hog_train = hog_images(x_train)
    x_hog_test = hog_images(x_test)
    sgdc = SGDClassifier(max_iter=50, loss='hinge')
    sgdc.fit(x_hog_train, y_train)
    print(classification_report(y_pred=sgdc.predict(x_hog_test), y_true=y_test))
    print(accuracy_score(sgdc.predict(x_hog_test), y_test))
