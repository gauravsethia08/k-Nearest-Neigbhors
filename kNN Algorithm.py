# -*- coding: utf-8 -*-
"""
Created on Fri May 17 01:29:43 2019

@author: Gaurav Sethia
"""

import pandas as pd
import numpy as np
import operator

data = pd.read_csv("C:\Gaurav\Project\Dataset\iris-species\Iris.csv")

def ed(data1, data2, length):
    
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
        
    return np.sqrt(distance)

def knn(ts, ti, k):
    distances = {}
    length = ti.shape[1]
     
    for x in range(len(ts)):
        dist = ed(ti, ts.iloc[x], length)
        distances[x] = dist[0]

    sorted_d = sorted(distances.items(), key = operator.itemgetter(1))
    ne = []
    
    for x in range(k):
        ne.append(sorted_d[x][0])
        
    cv = {}
    for x in range(len(ne)):
        r = ts.iloc[ne[x]][-1]
        
        if r in cv:
            cv[r] += 1
        
        else:
            cv[r] = 1
        
    sv = sorted(cv.items(), key = operator.itemgetter(1), reverse = True)
    return(sv[0][0], ne)
    
testset = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testset)

k = 3

res, n = knn(data, test, k)

print(res)
print(n)
    
