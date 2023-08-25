import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Path of image folders
from pathlib import Path

img_folder = Path('Case_3/GenderDataset/Gender database')
test_folder = img_folder / 'test'
train_folder = img_folder / 'train'

# File list
test_images = list(test_folder.glob('*.png'))
train_image = list(train_folder.glob('*.png'))

# Create training data and target
num_images = len(train_image)
X_train = np.empty((0,2700), dtype=float)
Y_train = np.empty(num_images, dtype=np.uint8) 

for i in range(num_images):
    image_path = str(train_image[i])
    # print(image_path)
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    # preprocessing

    # Normalize to range [0,1]
    img = img.astype(float)
    cv.normalize(img,img,0, 1.0, cv.NORM_MINMAX)
    # reshape to row vector
    img = np.reshape(img,(1,2700))
    X_train = np.vstack((X_train,img))
    # create Target vector
    if train_image[i].stem[0] == 'f':
        Y_train[i] = 0
    else:
        Y_train[i] = 1

# create MLP model
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(15,15), max_iter=500)
mlp.fit(X_train,Y_train)

#### print(X_train.shape)
#### print(Y_train.shape)
#### result = mlp.predict(X_train)

# Evaluation
from sklearn import metrics
# print('Accuracy = ', metrics.accuracy_score(Y_test,y_pred))
#### print('Accuracy = ', metrics.accuracy_score(Y_train,result))
metrics.plot_confusion_matrix(mlp, X_train,Y_train)
plt.show()

# Save model
import pickle
filename = 'GenderClassifier.sav'
pickle.dump(mlp, open(filename,'wb'))



