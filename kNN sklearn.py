# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:35:35 2019

@author: Gaurav Sethia
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

print(iris_data)
print(iris_label)

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.3)

classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


print("Confussion Martix is as follows ")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Matrix")
print(classification_report(y_test, y_pred))
