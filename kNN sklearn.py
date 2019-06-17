# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:35:35 2019

@author: Gaurav Sethia
"""

#importing libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets

#Loading dataset
iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target

#Printing Dataset
print(iris_data)
print(iris_label)

#Spliting the dataset for training and testing in 70:30 ratio
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.3)

#Creating a classifier with k = 5
classifier = KNeighborsClassifier(n_neighbors = 5)

#Fitting and building the model
classifier.fit(x_train, y_train)

#Predicting for test data
y_pred = classifier.predict(x_test)

#Printing Confusion Matrix
print("Confusion Martix is as follows ")
print(confusion_matrix(y_test, y_pred))

#Printing Accuracy Matrix
print("Accuracy Matrix")
print(classification_report(y_test, y_pred))
