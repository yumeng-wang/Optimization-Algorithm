#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:03:49 2020

@author: wangyumeng
"""


import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn import metrics


np.random.seed(1)

d = 1000
s = 20
n = 800
theta = np.zeros(d) 
theta[0:s] = 5
sigma = np.random.randn(n)
X = np.random.randn(n,d)
y = X.dot(theta) + sigma
lam_value=0.1


def coordinate_descent(y, X, lam_value, LN, times=10) :
    n,d = np.shape(X)
    w = np.zeros(d)
    for s in range(times) :
        for j in range(d) : 
            r = y - X.dot(w) + X[:,j]*w[j] 
            temp = 2*r.dot(X[:,j])/n - LN[j]
            sumj = 2*sum(X[:,j]**2)/n
            if temp < -lam_value:
                w[j] = (temp + lam_value)/sumj
            elif temp > lam_value:
                w[j] = (temp - lam_value)/sumj
            else :
                w[j] = 0
    return w

def cross_validation(y, X, lam_value, L, n_splits=3) :
    n,d = np.shape(X)
    KF = KFold(n_splits = n_splits)
    mse = np.zeros(n_splits)
    i = 0
    for train_index,test_index in KF.split(X):   
        w = coordinate_descent(y[train_index], X[train_index,:], lam_value, L)
        y_pred = X[test_index].dot(w)
        mse[i] = metrics.mean_squared_error(y[test_index], y_pred)
        i = i+1
    return np.mean(mse)

def find_lambda(y, X, lam_vector, L):   
    n,d = np.shape(X)
    MSE = np.zeros(len(lam_vector))
    for i in range(len(lam_vector)):
        MSE[i] = cross_validation(y, X, lam_vector[i], LN)
    lambd = lam_vector[np.where(MSE==np.min(MSE))]
    return lambd

lam = np.arange(0.1, 0.16, 0.02)
L = np.zeros(d)
l = find_lambda(y, X, lam, L)
print(l)
estimators = coordinate_descent(y, X, l, L).reshape(1,d)