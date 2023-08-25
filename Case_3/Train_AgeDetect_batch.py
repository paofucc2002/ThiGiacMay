import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import random

# Tao model MLP
print("Create MLP model")
mlp = MLPClassifier(hidden_layer_sizes=(256,128),activation='relu',max_iter=500,verbose=False)

# Tao list file hinh training
# Get list of training and test images
print("Loading data for training")
from pathlib import Path
img_folder = Path('Case_3/dataset/UTKface')  # GenderDataset
# test_folder = img_folder / 'testSet' / 'testSet'
train_folder = img_folder / 'Age10'
# test_images = list(test_folder.glob('*.jpg'))
train_images = list(train_folder.glob('**/*.jpg'))

print(train_images[0].parent.name) # lay ten folder
print(train_images[0].stem)        # lay ten file

# Doc tung file hinh -> tao X va Y
random.shuffle(train_images)
num_images = len(train_images)
print("number of images = ", num_images)
n_batches = 100
batch_size = int(num_images/n_batches)
print("batch size = ", batch_size)
for b in range(n_batches):
    print("Processing batch ", b+1)
    X = np.empty((0,40000),dtype = np.float32)
    Y = np.zeros((batch_size))
    for i in np.arange(batch_size):
        # load image
        image_path = str(train_images[b*batch_size + i])
        img = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
        # preprocessing: filter, rotate, change brightness...
        # normalize 
        img = img.astype(np.float32)
        cv.normalize(img, img, 0,1,cv.NORM_MINMAX)
        # reshape
        img = img.ravel()
        # stack images
        X = np.vstack((X,img))
        # create label
        Y[i] = int(train_images[i].parent.name)

    # Train MLP model
    print("Train MLP model with batch ", b+1)
    mlp = mlp.fit(X,Y)
    y_pred = mlp.predict(X)
    accuracy = metrics.accuracy_score(Y, y_pred)
    print('Batch ',b+1,' completed. Accuracy = ', accuracy)

print("Training completed")
# save model

import pickle
filename = 'Age.sav'
print("Save trained model as:",filename)
pickle.dump(mlp, open(filename, 'wb'))

# # Evaluation
# import matplotlib.pyplot as plt
# # print('Accuracy = ', metrics.accuracy_score(test_target,result))
# cm = metrics.confusion_matrix(Y, mlp.predict(X))
# disp = metrics.ConfusionMatrixDisplay(cm)
# disp.plot()
# plt.show()