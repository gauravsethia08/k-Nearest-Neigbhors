# -*- coding: utf-8 -*-
"""
Created on Sun May 19 01:35:20 2019

@author: Gaurav Sethia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
plt.style.use("ggplot")

data = pd.read_csv("C:\Gaurav\Robotics\Project\Dataset\iris-species\Iris.csv")

iris_data = data.drop("Species", axis = 1).values
iris_label = data["Species"].values

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size = 0.4)

n = np.arange(1,9)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))

for i,k in enumerate(n):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)
    
plt.title("KNN varying wih number of Neigbors")
plt.plot(n, test_accuracy, label="Test")
plt.plot(n, train_accuracy, label="Train")
plt.legend()
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.show()
