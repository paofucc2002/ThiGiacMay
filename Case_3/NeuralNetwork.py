import numpy as np
import cv2 as cv
from sklearn.neural_network import MLPClassifier

# Get list of training and test images
from pathlib import Path

img_folder = Path('Case_3/GenderDataset/Gender database') # Genader dataset
test_folder = img_folder / 'test'
train_folder = img_folder / 'train'

# File list
test_images = list(test_folder.glob('*.png'))
train_images = list(train_folder.glob('*.png'))

# Read every image -> convert to 1D array -> stack -> training data
num_images = len(train_images)
train_data = np.empty((0,2700), dtype=float)
train_target = np.empty(num_images, dtype=np.uint8) 
for i in range(num_images):
    # Load images
    image_path = str(train_images[i])
    img = cv.imread(str(image_path),cv.IMREAD_GRAYSCALE)

    # Preprocessing

    # Normalize
    img = img.astype(float)
    cv.normalize(img,img, 0,1,cv.NORM_MINMAX)

    # Reshape
    img = np.reshape(img,(1,2700))
    
    # Stack images.
    train_data = np.vstack((train_data,img))
    if train_images[i].stem[0]=='f':
        train_target[i] = 0
    else :
        train_target[i] = 1

# print(train_data.shape)
# print(train_target)
# Create MLP model
mlp = MLPClassifier(hidden_layer_sizes=(30,25),activation='relu',max_iter=500)

# Train MLP model
mlp = mlp.fit(train_data,train_target)

# Test
num_images = len(test_images)
test_data = np.empty((0,2700), dtype=float)
test_target = np.empty(num_images, dtype=np.uint8) 
for i in range(num_images):
    # Load images
    image_path = str(test_images[i])
    img = cv.imread(str(image_path),cv.IMREAD_GRAYSCALE)

    # Preprocessing

    # Normalize
    img= img.astype(float)
    cv.normalize(img,img, 0,1,cv.NORM_MINMAX)

    # Reshape
    img = np.reshape(img,(1,2700))
    
    # Stack images.
    test_data = np.vstack((test_data,img))
    if test_images[i].stem[0]=='f':
        test_target[i] = 0
    else :
        test_target[i] = 1


# print(train_data.shape)
# print(train_target.shape)
result1 = mlp.predict(train_data)
result2 = mlp.predict(test_data)

# Evaluation
import matplotlib.pyplot as plt
from sklearn import metrics
print('Train Accuracy = ', metrics.accuracy_score(train_target,result1))
print('Test Accuracy = ', metrics.accuracy_score(test_target,result2))
metrics.plot_confusion_matrix(mlp, train_data, train_target)
metrics.plot_confusion_matrix(mlp, test_data, test_target)
plt.show()