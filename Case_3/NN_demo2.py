import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from pathlib import Path
img_folder = Path(....)  # GenderDataset
test_folder = img_folder / 'test'
train_folder = img_folder / 'train'
test_images = list(test_folder.glob('*.png'))
train_images = list(train_folder.glob('*.png'))
# prepare training data
num_images = len(train_images)
train_data = np.empty((0,2700),dtype = np.float)
for i in range(num_images):
    # load image

    # preprocessing

    # normalize 

    # reshape

    # stack images


# build network