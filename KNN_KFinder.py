# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:35:20 2019

@author: Gaurav Sethia
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#changiing the plot style
plt.style.use("ggplot")

#Reading/Loading Dataset
data = pd.read_csv("Iris.csv") #Change the directory according to your system

#Separarting features and labels from the dataset
iris_data = data.drop("Species", axis = 1).values
iris_label = data["Species"].values

#Splitting the dataset for training and testing in 60:40 ratio
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.4)

#Making arrays of size n (here we are assuming that the optimal valu of k is less than 9)
n = np.arange(1,9)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))

#Predicting and checking accuracy for different values of k
for i,k in enumerate(n):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)

    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)
    
#Plotting a graph for find optimal k value
plt.title("KNN varying wih number of Neigbors")
plt.plot(n, test_accuracy, label="Test")
plt.plot(n, train_accuracy, label="Train")
plt.legend()
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show()
