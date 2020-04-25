#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:04:45 2020

@author: chang
"""

######################################
#
#Task 3
#
######################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

mu1, sigma1 = 0, 1 #mean and standard deviation
mu2, sigma2 = 3, 1
prior1, prior2 = 0.5, 0.5
N = 2000
data1 = np.random.normal(mu1, sigma1, int(N * prior1))
data2 = np.random.normal(mu2, sigma2, int(N * prior2))
data = np.concatenate((data1, data2))

mu10, sigma10 = 1, 1   #prior distribution
mu20, sigma20 = 1, 1

#plot the test data distribution
plt.hist(data, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()

def BPE_est(data, mu0, sigma0, sigma): # 1-dim
    n = data.shape[0]
    u_n_ave = np.sum(data) / n
    u_n = n*sigma0*sigma0 * u_n_ave / (n*sigma0*sigma0 + sigma*sigma) + sigma*sigma * mu0 / (n*sigma*sigma + sigma*sigma)
    
    sigma_n = (sigma0*sigma0 * sigma*sigma) / (n*sigma0*sigma0 + sigma*sigma)
    sigma_n += sigma*sigma
    return u_n, sigma_n
    
u1, sig1 = BPE_est(data1, mu10, sigma10, sigma1)
print('u1 = %f' % u1)
print('sigma1 = %f' % sig1)

u2, sig2 = BPE_est(data2, mu20, sigma20, sigma2)
print('u2 = %f' % u2)
print('sigma2 = %f' % sig2)

##############################2. number of sample VS parameter estimated

N1 = data1.shape[0]
N2 = data2.shape[0]
ratio1 = 0.8
ratio2 = 0.5
ratio3 = 0.1
u1_r1, sig1_r1 = BPE_est(data1[0:int(N1 * ratio1)], mu10, sigma10, sigma1)
u1_r2, sig1_r2 = BPE_est(data1[0:int(N1 * ratio2)], mu10, sigma10, sigma1)
u1_r3, sig1_r3 = BPE_est(data1[0:int(N1 * ratio3)], mu10, sigma10, sigma1)
print('Sample number of class 1 = %d, True u = %f, sigma = %f' % (N1, mu1, sigma1))
print('Use %d samples to estimate the parameter: u = %f, sigma = %f' %(int(N1*ratio1), u1_r1, sig1_r1))
print('Use %d samples to estimate the parameter: u = %f, sigma = %f' %(int(N1*ratio2), u1_r2, sig1_r2))
print('Use %d samples to estimate the parameter: u = %f, sigma = %f' %(int(N1*ratio3), u1_r3, sig1_r3))

N1 = data2.shape[0]
N2 = data2.shape[0]
ratio1 = 0.8
ratio2 = 0.5
ratio3 = 0.1
u2_r1, sig2_r1 = BPE_est(data2[0:int(N2 * ratio1)], mu20, sigma20, sigma2)
u2_r2, sig2_r2 = BPE_est(data2[0:int(N2 * ratio2)], mu20, sigma20, sigma2)
u2_r3, sig2_r3 = BPE_est(data2[0:int(N2 * ratio3)], mu20, sigma20, sigma2)
print('Sample number of class 2 = %d, True u = %f, sigma = %f' % (N2, mu2, sigma2))
print('Use %d samples to estimate the parameter: u = %f, sigma = %f' %(int(N2*ratio1), u2_r1, sig2_r1))
print('Use %d samples to estimate the parameter: u = %f, sigma = %f' %(int(N2*ratio2), u2_r2, sig2_r2))
print('Use %d samples to estimate the parameter: u = %f, sigma = %f' %(int(N2*ratio3), u2_r3, sig2_r3))

#############################3. classifier accuracy with estimated parameter

data1_new = np.random.normal(mu1, sigma1, int(N * prior1))
data2_new = np.random.normal(mu2, sigma2, int(N * prior2))
data_new = np.concatenate((data1_new, data2_new))

predict_label = np.zeros(N)
predict_label_para = np.zeros(N)
one = np.ones(int(N * prior1))
two = 2* np.ones(int(N * prior2))
true_label = np.concatenate((one, two))
for i in range(0,N):
    #print(i)
    x = data_new[i]
    #print(x)
    #BEYESIAN RULE
    post1 = prior1 * cal_p_dim1(mu1, sigma1, x)
    post2 = prior2 * cal_p_dim1(mu2, sigma2, x)
    # classification
    if post1 >= post2:
        predict_label[i] = 1
    else:
        predict_label[i] = 2
    
    # use estimated parameter to classifying
    post1_para = prior1 * cal_p_dim1(u1, sig1, x)
    post2_para = prior2 * cal_p_dim1(u2, sig2, x)
    if post1_para >= post2_para:
        predict_label_para[i] = 1
    else:
        predict_label_para[i] = 2

diff = true_label - predict_label
acc1 =   1 - np.count_nonzero(diff)  / N
print(acc1)
diff_para = true_label - predict_label_para
acc2= 1 - np.count_nonzero(diff_para) / N
print(acc2)

#########################4. number of sample VS the classification accuracy

predict_label = np.zeros(N)
predict_label_r1 = np.zeros(N)
predict_label_r2 = np.zeros(N)
predict_label_r3 = np.zeros(N)
one = np.ones(int(N * prior1))
two = 2* np.ones(int(N * prior2))
true_label = np.concatenate((one, two))
for i in range(0,N):
    x = data_new[i]
    
    # a. true parameter
    post1 = prior1 * cal_p_dim1(mu1, sigma1, x)
    post2 = prior2 * cal_p_dim1(mu2, sigma2, x)
    if post1 >= post2:
        predict_label[i] = 1
    else:
        predict_label[i] = 2
    
    # b. use all sample for parameter training
    post1_r1 = prior1 * cal_p_dim1(u1_r1, sig1_r1, x)
    post2_r1 = prior2 * cal_p_dim1(u2_r1, sig2_r1, x)
    if post1_r1 >= post2_r1:
        predict_label_r1[i] = 1
    else:
        predict_label_r1[i] = 2
        
    # c. use some sample for parameter training
    post1_r2 = prior1 * cal_p_dim1(u1_r2, sig1_r2, x)
    post2_r2 = prior2 * cal_p_dim1(u2_r2, sig2_r2, x)
    if post1_r2 >= post2_r2:
        predict_label_r2[i] = 1
    else:
        predict_label_r2[i] = 2
        
    # d. use fewer sample for parameter training
    post1_r3 = prior1 * cal_p_dim1(u1_r3, sig1_r3, x)
    post2_r3 = prior2 * cal_p_dim1(u2_r3, sig2_r3, x)
    if post1_r3 >= post2_r3:
        predict_label_r3[i] = 1
    else:
        predict_label_r3[i] = 2


diff = true_label - predict_label
acc =   1 - np.count_nonzero(diff)  / N
print('ACC based on true parameter: %f ' % acc)
diff_r1 = true_label - predict_label_r1
acc_r1= 1 - np.count_nonzero(diff_r1) / N
print('ACC based on parameter trained using %d samplesm : %f' % (int(N*ratio1), acc_r1))
diff_r2 = true_label - predict_label_r2
acc_r2 = 1 - np.count_nonzero(diff_r2) / N
print('ACC based on parameter trained using %d samplesm : %f' % (int(N*ratio2), acc_r2))
diff_r3 = true_label - predict_label_r3
acc_r3 = 1 - np.count_nonzero(diff_r3) / N
print('ACC based on parameter trained using %d samplesm : %f' % (int(N*ratio3), acc_r3))

