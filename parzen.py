#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE662 Mini Project 2

@author: chang

This is parzen window for task 1. 
Please run the data.py to generate the dataset before run this script.

"""

import numpy as np
from sklearn.metrics import  accuracy_score
from matplotlib import pyplot as plt


#------------------function-------------------

def in_hypercube(x, center, h):
    '''Detemines whether a point is within a hypercube
    This is also called kernal (smooth) function in the Density Estimation Field.
    Note: this function can handle the high dimensional situation.
    ARGUMENTS:
    x -- current data point
    center -- center point
    h -- hypercube unit length
    RETURNS:
    bool value  '''
    x_min = center - (0.5 * h)
    x_max = center + (0.5 * h)
    if (np.all(x_min < x) and np.all(x < x_max)):
        #print(x_min < x)
        #print(np.all(x_min < x))
        return True
    else:
        return False


def calculate_k(h, center, x_train):
    '''Calculate the number of training samples inside the hypercube
    ARGUMENTS:
    h -- integer, length of hypercube
    center -- coordinate of the center of the hypercube
    x_train: array of training data set
    RETURNS:
    k -- integer, number of training samples inside hypercube'''

    k = 0
    for x in x_train:
        is_inside = 0
        if in_hypercube(x, center, h):
            is_inside = 1
        k += is_inside
    return k


def estimate_density(h, x_train):
    '''Using parzen windows, estimates the probability density for the training set
    ARGUMENTS:
    h -- hypercube length
    x_train: training data set
    RETURNS:
    density -- array, estimated density of each training data sample'''
    n = len(x_train)
    d = len(x_train[0])
    c = 1 / (n * (h ** d))
    density = np.zeros((n, 1))
    for i, x in enumerate(x_train):
        x_density = c * calculate_k(h, x, x_train)
        density[i] = x_density
    return density

def classifier(density, x_train, x_test, y_train, h):
    yhat = np.zeros(y_train.shape)
    for i, x in enumerate(x_test):
        posterior = class_density(x, density, x_train, y_train, h)
        P = np.argmax(posterior)
        yhat[i] = P

    return yhat

def class_density(sample, density, x_train, y_train, h):
    classes = np.unique(y_train)
    posterior = np.zeros((len(classes), 1))
    for c in classes:
        p = 0
        x_temp = x_train[np.where(y_train == c)]
        density_temp = density[np.where(y_train == c)]
        for i, x in enumerate(x_temp):
            if in_hypercube(sample, x, h):
                p += density_temp[i]
        posterior[c] = p

    return posterior


def calculate_accuracy(y_test, yhat):
    '''Calculates the accuracy of the classifier
    ARGUMENTS:
    y_test -- ground truth labels
    yhat -- predicted labels
    RETURNS:
    score -- accuracy of prediction'''

    score = accuracy_score(y_test, yhat)
    return score

def plot_hist_dim1(x, y, bins):
    group1 = x[np.where(y == 0)]
    group2 = x[np.where(y == 1)]
    
    bins = np.linspace(x.min(), x.max(), len(x)/2)
    plt.hist(group1, bins, alpha=0.5, label='x')
    plt.hist(group2, bins, alpha=0.5, label='y')
    plt.legend(loc='upper right')
    plt.show()
    
    


#----------------Task1 main-------------------
h_size = np.array([0.1, 0.5, 1, 2, 5, 10])
k = 0
for (key1, x_train), (key2, x_test), (key3, y_train), (key4, y_test) in \
         zip(X_train.items(), X_test.items(), Y_train.items(), Y_test.items()):
    print("current dataset ::: ", key1)
    
    acc = []
    k = k+1
    for h_tmp in h_size:
        print(h_tmp)
        density_est = estimate_density(h_tmp, x_train)
        Yhat = classifier(density_est, x_train, x_test, y_train, h_tmp)
        acc.append(calculate_accuracy(y_test, Yhat))
        #print("acc = ", acc)
    
    plt.figure()
    x = np.array([0,1,2,3,4,5])
    y = acc
    plt.plot(x,y,".-")
    plt.ylabel("accuracy")
    plt.xlabel("window size (h)")
    plt.xticks(x,['0.1','0.5','1','2','5','10'])
    plt.show()  
        
    if(k == 2):
        break
    
    
    
    plot_hist_dim1(X_train["c2_s80_n1"], Y_train["c2_s80_n1"], bins=int(40/10))




