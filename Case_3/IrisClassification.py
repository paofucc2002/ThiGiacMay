import numpy as np
import matplotlib.pyplot as plt
# Adaboost library
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics # calculate the accuracy
import pickle

# Load dataset
iris = datasets.load_iris()
X = iris.data    # feature values
Y = iris.target  # labels
# split data: 30% test, 70% training
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)

# create adaboost model
ada = AdaBoostClassifier(n_estimators=30, learning_rate=1)
# train classifier
model = ada.fit(X_train,Y_train)

# use classifier
y_pred = model.predict(X_test)
print('accuracy = ', metrics.accuracy_score(Y_test, y_pred))

# save model
filename = 'Iris_classifier.sav'
pickle.dump(model, open(filename, 'wb'))
