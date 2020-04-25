#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:39:46 2020

@author: chang
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#--------------question 3--------------
mean1 = (1, 2)
cov1= [[0.5, 0], [0, 0.5]]
x1 = np.random.multivariate_normal(mean1, cov1, 30)
x1.shape
x1_label = np.zeros((30),dtype=int)


mean2 = (2, 1)
x2 = np.random.multivariate_normal(mean2, cov1, 30)
x2.shape
x2_label = np.ones((30),dtype=int)

mean3 = (3, 3)
x3 = np.random.multivariate_normal(mean3, cov1, 30)
x2.shape
x3_label = np.ones((30),dtype=int) *2

mean4 = (4, 2)
x4 = np.random.multivariate_normal(mean4, cov1, 30)
x4_label = np.ones((30),dtype=int) *3

mean5 = (-1, -1)
x5 = np.random.multivariate_normal(mean5, cov1, 30)
x5_label = np.ones((30),dtype=int) *4


x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
label_true = np.concatenate((x1_label, x2_label, x3_label, x4_label, x5_label), axis=0)



#kmeans

plt.figure(figsize=(8, 12))
plt.subplot(221)
plt.scatter(x[:, 0], x[:, 1], c=label_true, alpha=1)
plt.title("True Label")


plt.subplot(222)
y_pred = KMeans(n_clusters=4, random_state=0).fit(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred.labels_,  alpha=1)
plt.title("k=4 kmeans results")
plt.scatter(y_pred.cluster_centers_[:, 0], y_pred.cluster_centers_[:, 1], c='red', marker='x')
plt.show()


plt.subplot(223)
y_pred3 = KMeans(n_clusters=5, random_state=0).fit(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred3.labels_,  alpha=1)
plt.title("k=5 kmeans results")
plt.scatter(y_pred3.cluster_centers_[:, 0], y_pred3.cluster_centers_[:, 1], c='red', marker='x')
plt.show()

plt.subplot(224)
y_pred3 = KMeans(n_clusters=6, random_state=0).fit(x)
plt.scatter(x[:, 0], x[:, 1], c=y_pred3.labels_,  alpha=1)
plt.title("k=6 kmeans results")
plt.scatter(y_pred3.cluster_centers_[:, 0], y_pred3.cluster_centers_[:, 1], c='red', marker='x')
plt.show()

