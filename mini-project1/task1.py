#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:22:21 2020

@author: chang
"""


######################################
#
#Task 1
#
######################################
# 1-dim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

mu1, sigma1 = -1, 1 #mean and standard deviation
mu2, sigma2 = 1, 1
prior1, prior2 = 0.5, 0.5
N = 1000
data1 = np.random.normal(mu1, sigma1, int(N * prior1))
data2 = np.random.normal(mu2, sigma2, int(N * prior2))
data = np.concatenate((data1, data2))
#plot the test data distribution
plt.hist(data, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()

#function to calculate the likelihood for high dimension data
def cal_p_dim1(mu, sigma, x):
    p = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * np.square((x-mu)/sigma))
    return p

predict_label = np.zeros(N)
one = np.ones(int(N * prior1))
two = 2* np.ones(int(N * prior2))
true_label = np.concatenate((one, two))
for i in range(0,N):
    #print(i)
    x = data[i]
    #print(x)
    #BEYESIAN RULE
    post1 = prior1 * cal_p_dim1(mu1, sigma1, x)
    post2 = prior2 * cal_p_dim1(mu2, sigma2, x)
    #post1 = 1/(sigma1 * np.sqrt(2*np.pi)) * prior1 * np.exp(-0.5 * np.square((x-mu1)/sigma1))
    #post2 = 1/(sigma2 * np.sqrt(2*np.pi)) * prior2 * np.exp(-0.5 * np.square((x-mu2)/sigma2))
    # classification
    if post1 >= post2:
        predict_label[i] = 1
    else:
        predict_label[i] = 2

diff = true_label - predict_label
acc1 =   1 - np.count_nonzero(diff)  / N
print(acc1)


# 2 dimension
dim = 2
#u1 = np.ones(2) * (-1)
u1 = np.array([-1,1])
u2 = np.ones(2) * 1
prior = np.array([0.5,0.5])
cov1 = np.array(([1,0],[0,1]))
cov2 = np.array(([1,0],[0,1]))
data1 = np.random.multivariate_normal(u1, cov1, int(N * prior[[0]]))
data2 = np.random.multivariate_normal(u2, cov2, int(N * prior[[1]]))
data = np.concatenate((data1, data2))

#function to calculate the likelihood for high dimension data
def cal_p(u, cov, xx):
    k = u.shape[0]
    fir = 1 / np.sqrt( np.power(2*np.pi, k) * np.linalg.det(cov))
    dis = xx - u
    power = -0.5 * dis[np.newaxis,:] @ np.linalg.inv(cov) @ dis[:,np.newaxis]
    second = np.exp(power)
    return fir*second

predict_label = np.zeros(N)
one = np.ones(int(N * prior1))
two = 2* np.ones(int(N * prior2))
true_label = np.concatenate((one, two))
for i in range(0,N):
    x = data[i,:]
    #print(x)
    post1 = prior1 * cal_p(u1, cov1, x)
    post2 = prior2 * cal_p(u2, cov2, x)
    if post1 >= post2:
        predict_label[i] = 1
    else:
        predict_label[i] = 2
diff = true_label - predict_label
acc2 =   1 - np.count_nonzero(diff)  / N
print(acc2)


#scatter plot of the data (true label)
color1 = ['red'] * int(N*prior[[0]])
color2 = ['blue']* int(N*prior[[1]])
color = color1 + color2
plt.subplot(1,2,1)
plt.scatter(data[:,0], data[:,1] ,c=color, alpha=0.4)
plt.title('original class label')
plt.xlabel('dim 1'); plt.ylabel('dim 2')
#plot predicted points
color_pred = color
for i in range(0,N):
    if predict_label[i] == 1:
        color_pred[i] = "red"
    else:
        color_pred[i] = "blue"
plt.subplot(1,2,2)
plt.scatter(data[:,0], data[:,1] ,c=color_pred, alpha=0.4)
plt.title('predicted class label')
plt.xlabel('dim 1'); plt.ylabel('dim 2')


# 3 dimension
dim = 3
#u1 = np.ones(2) * (-1)
u1 = np.array([-1,1,1])
u2 = np.ones(3) * 1
prior = np.array([0.5,0.5])
cov1 = np.array(([1,0,0],[0,1,0],[0,0,1]))
cov2 = np.array(([1,0,0],[0,1,0],[0,0,1]))
data1 = np.random.multivariate_normal(u1, cov1, int(N * prior[[0]]))
data2 = np.random.multivariate_normal(u2, cov2, int(N * prior[[1]]))
data = np.concatenate((data1, data2))


predict_label = np.zeros(N)
one = np.ones(int(N * prior1))
two = 2* np.ones(int(N * prior2))
true_label = np.concatenate((one, two))
for i in range(0,N):
    x = data[i,:]
    #print(x)
    post1 = prior1 * cal_p(u1, cov1, x)
    post2 = prior2 * cal_p(u2, cov2, x)
    if post1 >= post2:
        predict_label[i] = 1
    else:
        predict_label[i] = 2
diff = true_label - predict_label
acc2 =   1 - np.count_nonzero(diff)  / N
print(acc2)


#scatter plot of the data (true label)
color1 = ['red'] * int(N*prior[[0]])
color2 = ['blue']* int(N*prior[[1]])
color = color1 + color2
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=color, alpha=0.4)
ax.set_xlabel('dim 1');ax.set_ylabel('dim 2'); ax.set_zlabel('dim 3')
ax.set_title('original class label')

#plot predicted points
color_pred = color
for i in range(0,N):
    if predict_label[i] == 1:
        color_pred[i] = "red"
    else:
        color_pred[i] = "blue"
bx = fig.add_subplot(122, projection='3d')
bx.scatter(data[:,0], data[:,1], data[:,2], c=color_pred, alpha=0.4)
bx.set_xlabel('dim 1');bx.set_ylabel('dim 2'); bx.set_zlabel('dim 3')
bx.set_title('predicted class label')











