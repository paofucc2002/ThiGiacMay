#import modules
import pandas as pd
import numpy as np
# load adaboost library
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

#read csv to pandas dataframe
mushrooms = pd.read_csv("Case_3/dataset/mushrooms.csv")

#create dummy variables
mushrooms = pd.get_dummies(mushrooms)

#subset data into dependent and independent variables x,y
LABELS = ['class_e', 'class_p']
FEATURES = [a  for a in mushrooms.columns if a not in LABELS ]
y = mushrooms[LABELS[0]]
x= mushrooms[FEATURES]

# split data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3)

ada = AdaBoostClassifier(n_estimators=20, learning_rate=1)
# train classifier
ada.fit(X_train,Y_train)


# save model
import pickle
filename = 'detector.sav'
pickle.dump(ada, open(filename,'wb'))



# create and train the classifier
number_of_learners = 50
fig = plt.figure(figsize=(10,10))
ax0 =  fig.add_subplot(111)

accuracy = np.empty(number_of_learners, dtype=float)
for i in np.arange(1,number_of_learners+1):
    ada = AdaBoostClassifier(n_estimators=i, learning_rate=1)
    # train classifier
    ada.fit(X_train,Y_train)
    y_pred = ada.predict(X_test)
    accuracy[i-1] = metrics.accuracy_score(Y_test,y_pred)

ax0.plot(np.arange(1,number_of_learners+1), accuracy)
ax0.set_xlabel('# of weak leaners')
ax0.set_ylabel('Accuracy')

plt.show()