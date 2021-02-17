import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

'''
Univariate linear regression
'''
x = np.arange(0,4)
y = np.arange(0,4)

# hypothesis
def h(x, theta0, theta1):
    return theta0 + theta1 * x

print(np.sum((y-h(x, 0, 0.2))**2))

# cost function
def J(theta0, theta1):
    return 1.0/2.0/x.shape[0] * np.sum((y-h(x, theta0, theta1))**2)

fig, ax = plt.subplots(1, 2)
ax[0].set_xlim(0,4)
ax[0].set_ylim(0,4)
ax[1].set_xlim(0,4)
ax[1].set_ylim(0,4)
# print(np.arange(0.5, 2, 0.5))

for theta1 in np.arange(0.0, 2, 0.1):
    ax[0].plot(x, h(x, 0, theta1))
    ax[1].plot(theta1, J(0, theta1), "o")

plt.show()




'''
scikit
'''


# sample data iris
iris = datasets.load_iris()
dir(iris)
iris.feature_names
iris.data
iris.target_names
iris.target

# sample data digits
digits = datasets.load_digits()
dir(digits)
digits.target

digits.target.size # n examples
digits.data.shape[1] # n features


###
# support vector machine - estimator
###
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.) # Parameter C, gamma manuell gesetzt

# lernen
X = digits.data[:-1]
X
y = digits.target[:-1]
y
clf.fit(X, y)

# vorhersagen
clf.predict(digits.data[-1:])


###
# k nearest neighbor für den Iris-Datensatz
###
# Split iris data in train and test data
# A random permutation, to split the data randomly
iris_X = iris.data
iris_y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
indices


iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
>>> iris_X_test = iris_X[indices[-10:]]
>>> iris_y_test = iris_y[indices[-10:]]
>>> # Create and fit a nearest-neighbor classifier
>>> from sklearn.neighbors import KNeighborsClassifier
>>> knn = KNeighborsClassifier()
>>> knn.fit(iris_X_train, iris_y_train) 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=5, p=2,
           weights='uniform')
>>> knn.predict(iris_X_test)
array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
>>> iris_y_test
array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])




