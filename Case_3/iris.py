import pickle
import numpy
from sklearn.ensemble import AdaBoostClassifier

model = pickle.load(open('Iris_classifier.sav','rb'))
y_pred = model.predict(data)