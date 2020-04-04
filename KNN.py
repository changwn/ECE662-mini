#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:58:26 2020

@author: chang
"""

import numpy as np
from scipy.spatial import distance as disLib
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

#----------------function--------------------

def k_value(s):
    '''Determines the k values to use according to number of samples in the data set
    ARGUMENTS:
    s -- integer, number of samples in data set
    RETURNS:
    k -- list containing all k values to try on the data set
    '''
    if (s <= 100):
        k = [1, 3, 5, 7, 9]
    elif (100 < s <= 500):
        k = [1, 3, 5, 7, 9]
    elif (500 < s <= 2000):
        k = [1, 3, 5, 7, 9]
    elif (2000 < s <= 5000):
        k = [1, 3, 5, 7, 9]
    else:
        k = [1, 3, 5, 7, 9]
    return k

def calculate_accuracy(y_test, yhat):
    '''Calculates the accuracy of the classifier
    ARGUMENTS:
    y_test -- ground truth labels
    yhat -- predicted labels
    RETURNS:
    score -- accuracy of prediction'''

    score = accuracy_score(y_test, yhat)
    return score

def euclidean(sample_tr, sample_te):
    '''
    Calculates the L2 normalization, which is also the euclidean distance, between two samples
    ARGUMENTS:
    sample_tr -- training sample
    sample_te -- testing sample
    RETURNS:
    dist -- euclidean distance between sample_tr and sample_te
    '''
    dist = np.linalg.norm(sample_tr - sample_te)
    return dist

def manhattan(sample_tr, sample_te):
    '''
    Calculates the manhattan distance between two samples
    ARGUMENTS:
    sample_tr -- training sample
    sample_te -- testing sample
    RETURNS:
    dist -- manhattan distance between sample_tr and sample_te
    '''
    dist = disLib.cityblock(sample_tr, sample_te)
    return dist

def k_neighbors(x_train, y_train, sample, k, distance):
    '''Finds the distance between all points in the training set and the sample point to find the
    top k closest neighbors
    ARGUMENTS:
    x_train -- training set
    y_train -- training set ground truth label
    sample -- sample point
    k -- integer, number of neighbors wanted
    distance -- manhattan or euclidean, distance calculation function
    RETURNS:
    neighbors -- array with top k closest neighbors to sample point'''
    dist = np.zeros(x_train.shape)
    for i, x in enumerate(x_train):
        d = distance(x, sample)
        dist[i] = d

    sort = np.argsort(dist, axis = 0)
    sort = sort.flatten()
    neighbors = y_train[sort[:k]]
    return neighbors

def classifier(x_train, y_train, sample, k, distance):
    '''Finds the class with the most neighbors to the sample point
    ARGUMENTS:
    x_train -- training data set
    y_train -- training data set ground truth labels
    sample -- sample point
    k -- integer, k value for KNN
    distance -- manhattan or euclidean, distance calculation function
    RETURNS:
    max -- class with the most neighbors with the sample point'''
    neighbors = k_neighbors(x_train, y_train, sample, k, distance)
    counts = np.bincount(neighbors.flatten())
    max = np.argmax(counts)

    return max

def predict(x_train, y_train, x_test, k, distance):
    '''
    Predicts label for testing set
    ARGUMENTS:
    x_train -- training data set
    y_train -- training data set ground truth labels
    x_test -- testing data set
    k -- integer, k value for KNN
    distance -- manhattan or euclidean, distance calculation function
    RETURNS:
    yhat -- predicted labels for testing set
    '''
    yhat = np.zeros(y_train.shape)
    for i, x in enumerate(x_test):
        p = classifier(x_train, y_train, x, k, distance)
        yhat[i] = p

    return yhat

def evaluate(x_train, x_test, y_train, y_test, distance):
    '''Runs and evaluates a KNN classifier using training and testing data
    ARGUMENTS:
    x_train -- training set
    x_test -- testing set
    y_train -- training set ground truth labels
    y_test -- testing set ground truth labels
    distance -- manhattan or euclidean, distance calculation function
    RETURNS:
    yhat -- predicted labels
    score -- accuracy score of prediction
    '''
    yhat = {}
    score = {}
    data_size = len(x_train)
    K = k_value(data_size)
    for k in K:
        prediction = predict(x_train, y_train, x_test, k, distance)
        accuracy = calculate_accuracy(y_test, prediction)
        yhat.update({k: prediction})
        score.update({k: accuracy})

    return yhat, score

def plot_accuracy(accuracy):
    
    pass





#--------------task2 main--------------------
#analyzing the effects of varying K and N on the task-defined class and sample size using euclidean distance
Yhat = {}
Accuracy = {}

for key in X_train.keys():
    yhat, accuracy = evaluate(X_train[key], X_test[key], Y_train[key], Y_test[key], euclidean)
    print(key)
    Yhat.update({key: yhat})
    Accuracy.update({key: accuracy})

#plot
for key in X_train.keys():    
    plt.figure()
    x = np.array([1,3,5,7,9])
    y = Accuracy[key].values()
    plt.plot(x,y,".-")
    plt.title(key)
    plt.ylabel("accuracy")
    plt.xlabel("K nearest neighbors (k)")
    plt.xticks(x,['1','3','5','7','9'])
    #plt.show()  
    plt.savefig("knn_" + key + ".png")

#manhattan distance
Yhat = {}
Accuracy = {}

wish_list = ['c2_s80_n1','c2_s1000_n1','c5_s80_n2']
for key in X_train.keys():
    if key in wish_list:
        yhat, accuracy = evaluate(X_train[key], X_test[key], Y_train[key], Y_test[key], manhattan)
        print(key)
        Yhat.update({key: yhat})
        Accuracy.update({key: accuracy})

#plot
for key in X_train.keys():
    if key in wish_list:
        plt.figure()
        x = np.array([1,3,5,7,9])
        y = Accuracy[key].values()
        plt.plot(x,y,".-")
        plt.title(key+ "Manhattan Dist")
        plt.ylabel("accuracy")
        plt.xlabel("K nearest neighbors (k)")
        plt.xticks(x,['1','3','5','7','9'])
        #plt.show()  
        plt.savefig("man_" + key + ".png")        

    
    
    
    