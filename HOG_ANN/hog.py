from skimage.feature import hog
from skimage import color
import numpy as np


def apply_hog(image):
    image = color.rgb2gray(image)
    hog_image = hog(image, orientations=9, pixels_per_cell=(
        4, 4), cells_per_block=(2, 2), transform_sqrt=True)
    return hog_image


def hog_images(images):
    return np.vstack([apply_hog(x) for x in images])
