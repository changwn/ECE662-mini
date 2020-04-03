#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE662 Mini Project 2

@author: chang


This script use to generate the data. 

"""

import random
import numpy as np

def data_parameters(type):
    '''
    Initializes desired parameters for data generation
    ARGUMENTS:
    type -- string of the following possibilities:
            "set" -- use preset parameters by problem
            "vary" -- user defined parameters for the problem
    RETURNS:
    classes -- array containing lists of # of classes for each subset of data
    samples -- array containing lists of # of samples for each class for each subset of data
    dim -- array containing lists of # of dimensions for each feature vector for each subset of data
    '''
    if type == "set":
        classes = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [5]])
        samples = np.array([[80, 80, 80, 80], [1000, 1000, 1000, 1000], [80]])
        dims = np.array([[1, 2, 5, 10], [1, 2, 5, 10], [2]])
    elif type == "vary":
        pass

    #asserting the right number of parameters are being returned
    assert len(classes[0]) == len(samples[0]) == len(dims[0]), \
        "first set not equal length"
    assert len(classes[1]) == len(samples[1]) == len(dims[1]), \
        "second set not equal length"
    assert len(classes[2]) == len(samples[2]) == len(dims[2]), \
        "third set not equal length"
    assert len(classes) == len(samples) == len(dims), \
        "all data parameters must be matching sizes"

    return classes, samples, dims

def generate_data(type):
    '''
    Randomly generates data of normal distribution with each class having the same priors
    ARGUMENTS:
    type -- string of the following possibilities:
        "set" -- use preset parameters by problem
        "vary" -- parameters set by me
    RETURNS:
    X_train -- dictionary of training samples sorted by keys in the format "class#_sample#_dim#" (ex: c2_s80_n1)
    X_test -- dictionary of testing samples sorted by keys in the format "class#_sample#_dim#" (ex: c2_s80_n1)
    Y_train -- dictionary of ground truth labels of the training sets sorted by keys in the format "class#_sample#_dim#" (ex: c2_s80_n1)
    Y_test -- dictionary of ground truth labels of the testing sets sorted by keys in the format "class#_sample#_dim#" (ex: c2_s80_n1)
    '''
    X_train = {}
    X_test = {}
    Y_train = {}
    Y_test = {}
    classes, samples, dims = data_parameters("set")
    for i in range(len(classes)):
        for (c, s, d) in zip(classes[i], samples[i], dims[i]):
            print(c, s, d)
            mu = generate_mu(c, d)
            sigma = generate_sigma(c, d)
            x_train_temp = []
            x_test_temp = []
            y_train_temp = []
            y_test_temp = []
            for j in range(c):
                #univariate gaussian
                if d == 1:
                    x_train = np.random.normal(mu[j], sigma[j], (int(s/2), d))
                    x_test = np.random.normal(mu[j], sigma[j], (int(s/2), d))
                    y_train = np.full((int(s/2), 1), j)
                    y_test = np.full((int(s/2), 1), j)

                    x_train_temp.append(x_train)
                    x_test_temp.append(x_test)
                    y_train_temp.append(y_train)
                    y_test_temp.append(y_test)
                #multivariate gaussian
                else:
                    x_train = np.random.multivariate_normal(mu[j], sigma[j], int(s/2))
                    x_test = np.random.multivariate_normal(mu[j], sigma[j], int(s/2))
                    y_train = np.full((int(s/2), 1), j)
                    y_test = np.full((int(s/2), 1), j)

                    x_train_temp.append(x_train)
                    x_test_temp.append(x_test)
                    y_train_temp.append(y_train)
                    y_test_temp.append(y_test)

            X_train.update({"c{}_s{}_n{}".format(c, s, d): np.concatenate(x_train_temp)})
            X_test.update({"c{}_s{}_n{}".format(c, s, d): np.concatenate(x_test_temp)})
            Y_train.update({"c{}_s{}_n{}".format(c, s, d): np.concatenate(y_train_temp)})
            Y_test.update({"c{}_s{}_n{}".format(c, s, d): np.concatenate(y_test_temp)})

    return X_train, X_test, Y_train, Y_test

def generate_mu(classes, dims):
    '''
    Randomly generates mu values for all sub-datasets
    ARGUMENTS:
    classes -- number of classes in this subset data
    dims -- dimension of feature vector
    RETURNS:
    mu -- array containing mu values
    '''
    mu = np.zeros((classes, dims))
#    for m in mu:
#        for i in range(len(m)):
#            #range for mu is 0.0 to ((# of classes) * constant), constant = 4
#            m[i] = round(random.uniform(0.0, classes * 4.0), 2)
#    return mu
    for i in range(classes):
        for j in range(dims):
            mu[i,j] = 1 + i*2   #distence between u is 2
    return mu

def generate_sigma(classes, dims):
    '''
    Randomly generates sigma value for all sub-datasets
    ARGUMENTS:
    classes -- number of classes in this subset data
    dims -- dimension of feature vector
    RETURNS:
    sigma -- array containing sigma values
    '''
    sigma = np.zeros((classes, dims, dims))
#    for s in sigma:
#        for i in range(len(s)):
#            #range for mu is 0.0 to ((# of classes) * constant * 3/4), constant = 4
#            s[i][i] = round(random.uniform(0.0, classes * 3.0), 2)
#    return sigma
    for i in range(classes):
        for j in range(dims):
            sigma[i, j, j] = 1
    return sigma



#################################################################
X_train, X_test, Y_train, Y_test = generate_data("set")


