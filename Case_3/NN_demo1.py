import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load dataset
digit = datasets.load_digits()
X = digit['data'] 
print(X.shape)

Y = digit['target']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

# build Neural network model
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30,30),max_iter=500)
mlp.fit(X_train,Y_train)

y_predict = mlp.predict(X_test)
print(metrics.accuracy_score(Y_test,y_predict))
metrics.plot_confusion_matrix(mlp, X_test,Y_test)
plt.show()

