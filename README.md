# k-Nearest-Neigbhors
kNN is one of many supervised learning algorithms, used in data mining and machine learning. Its is classifier algorithm where the learning in based on how similar is a data from other. 
kNN can be used for both classification and regression predective problems.

# Pseduo Code
1) Load the data
2) Initialize the value of k
3) For getting the predicted classes, iterate from 1 to total numer of training data point
      (i)   Calculate distance between test data and each row of training data.
      (ii)  Sort the calculated distances i ascending order
      (iii) Get the top k rows of the sorted array
      (iv)  Get the most frequent class of these rows
      (v)   Return the predicted class
      
# Pros 
1) No assumption about data - useful for non linear data.
2) Simple Algortihm
3) High Accuracy
4) Veratile

# Cons
1) Computationally expensive - because the algorithm stores all the training data.
2) High memory requirement.
3) Prediction stage might be slow
4) Doesnt works well with large number of features

# Required Libraries
1) sklearn
2) pandas
3) numpy
