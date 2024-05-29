#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 23:58:59 2019

@author: Christian
"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


## sample dat iris (classify iris type)
iris = datasets.load_iris()
iris.feature_names
iris.data
type(iris.data)
iris.data.shape
iris.target_names
iris.target



## sample dat digits (classify digit)
digits = datasets.load_digits()
digits.target
digits.target.size # n examples
digits.data.shape[1] # n features




## support vector machine - estimator
clf = svm.SVC(gamma=0.001, C=100.) # Parameter C, gamma manuell gesetzt
X = digits.data[:-1]
y = digits.target[:-1]
clf.fit(X, y)
clf.predict(digits.data[-1:])

# split into traingin and test dat
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)          

sorted(sk.metrics.SCORERS.keys())
# k-fold CV
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
scores                                              
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='neg_mean_absolute_error')
scores


# k nearest neighbor für den Iris-Datensatz
iris_X = iris.data
iris_y = iris.target
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 
knn.predict(iris_X_test)
iris_y_test


## Data transformation with held out dat






## Pipelines

# ColumnTransformer class to bundle together different preprocessing steps



















##

# support vector machine 
model_smv = svm.SVC(gamma='scale') 
yt = y_train.iloc[:,4]
yv = y_val.iloc[:,4]
model_smv.fit(X_train, yt)
predictions = model_smv.predict(X_val)
mean_absolute_error(predictions, yv)
# meh: 13.92


