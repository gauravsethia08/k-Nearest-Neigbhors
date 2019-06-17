# -*- coding: utf-8 -*-
"""
Created on Fri May 17 01:29:43 2019

@author: Gaurav Sethia
"""
#importig libraries
import pandas as pd
import numpy as np
import operator

#reading/loading the dataset
data = pd.read_csv("Iris.csv") #change the directory according to your system

#function to calculate euclidean distance
def euclidean_distance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

#kNN Algorithmic Code
def knn(ts, ti, k):
    distances = {}
    length = ti.shape[1]
     
    #Calculating Distances all of the data points
    for x in range(len(ts)):
        dist = euclidean_distance(ti, ts.iloc[x], length)
        distances[x] = dist[0]

    #Sorting the distances calculated
    sorted_d = sorted(distances.items(), key = operator.itemgetter(1))
    ne = []
    
    #Taking top k values
    for x in range(k):
        ne.append(sorted_d[x][0])
        
    #Counting the Number of times index appears in new list
    cv = {}
    for x in range(len(ne)):
        r = ts.iloc[ne[x]][-1]
        if r in cv:
            cv[r] += 1
        else:
            cv[r] = 1
        
    #Sorting in ascending order to determine the closest neighbor
    sv = sorted(cv.items(), key = operator.itemgetter(1), reverse = True)
    return(sv[0][0], ne)
    
#Entering Some Random Values for prediction
testset = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testset)

#k Value
k = 3

#Predicting Step
res, n = knn(data, test, k)

#Printing Results
print(res)
print(n)
