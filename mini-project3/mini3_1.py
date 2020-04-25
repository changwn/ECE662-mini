#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:39:46 2020

@author: chang
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#--------------question 1--------------
mean1 = (1, 2)
cov1= [[1, 0], [0, 1]]
x1 = np.random.multivariate_normal(mean1, cov1, 30)
x1.shape
x1_label = np.ones((30),dtype=int)


mean2 = (2, 1)
cov2 = [[1, 0], [0, 1]]
x2 = np.random.multivariate_normal(mean2, cov2, 30)
x2.shape
x2_label = np.zeros((30),dtype=int)

x = np.concatenate((x1, x2), axis=0)
label_true = np.concatenate((x1_label, x2_label), axis=0)



plt.figure(figsize=(8, 12))
plt.subplot(221)
plt.scatter(x[:, 0], x[:, 1], c=label_true)
plt.title("True Label")


plt.subplot(222)
y_pred = KMeans(n_clusters=2, random_state=0).fit(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred.labels_)
plt.title("k=2 kmeans results")
plt.scatter(y_pred.cluster_centers_[:, 0], y_pred.cluster_centers_[:, 1], c='red', marker='x')
plt.show()


plt.subplot(223)
y_pred3 = KMeans(n_clusters=3, random_state=0).fit(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred3.labels_)
plt.title("k=3 kmeans results")
plt.scatter(y_pred3.cluster_centers_[:, 0], y_pred3.cluster_centers_[:, 1], c='red', marker='x')
plt.show()


