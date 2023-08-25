import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Path of image folders
import random
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Tao model MLP
print("Create MLP model")
mlp = MLPClassifier(hidden_layer_sizes=(128,64),activation='relu',max_iter=1000,verbose=False)

img_folder = Path('Case_3/dataset/MNIST')
test_folder = img_folder / 'testSet' / 'testSet'
train_folder = img_folder / 'trainingSet' / 'trainingSet'
test_image = list(test_folder.glob('*.jpg'))
train_image = list(train_folder.glob('**/*.jpg'))

print(len(train_image))
print(len(test_image))

# print(train_image[3].parent.name) # lay ten folder
# print(train_image[3].stem)        # lay ten file

# Create training data and target
# Doc tung file hinh -> tao X va Y
random.shuffle(train_image)
num_images = len(train_image)
# print("number of images = ", num_images)
n_batches = 10
batch_size = int(num_images/n_batches)
# print("batch size = ", batch_size)


for b in range(n_batches):
    print("Processing batch ", b+1)
    X_train = np.empty((0,784), dtype=float)
    Y_train = np.empty(batch_size, dtype=np.uint8) 
    for i in np.arange(batch_size):
        image_path = str(train_image[b*batch_size + i])
        # print(image_path)
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        # preprocessing

        # Normalize to range [0,1]
        img = img.astype(float)
        cv.normalize(img,img,0, 1, cv.NORM_MINMAX)
        # reshape to row vector
        img = np.reshape(img,(1,784))
        X_train = np.vstack((X_train,img))
        # create label
        Y_train[i] = int(train_image[i].parent.name)
    
    # Train MLP model
    print("Train MLP model with batch ", b+1)
    mlp = mlp.fit(X_train,Y_train)
    y_pred = mlp.predict(X_train)
    accuracy = metrics.accuracy_score(Y_train, y_pred)
    print('Batch ',b+1,' completed. Accuracy = ', accuracy)

#### print(X_train.shape)
#### print(Y_train.shape)
# result = mlp.predict(X_train)

print("Training Complete")

# Save model
import pickle
filename = 'MNISTsave.sav'
print("Save trained model as:",filename)
pickle.dump(mlp, open(filename,'wb'))

# Evaluation
# print('Accuracy = ', metrics.accuracy_score(Y_test,y_pred))
# print('Accuracy = ', metrics.accuracy_score(Y_train,result))
metrics.plot_confusion_matrix(mlp, X_train,Y_train)
plt.show()