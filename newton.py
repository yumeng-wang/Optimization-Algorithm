#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:47:02 2020

@author: wangyumeng
"""

import torch
from torch.autograd import Variable
torch.manual_seed(1)

def Grad(L, x):
    # calculate gradient
    grad = torch.autograd.grad(L(x), x)
    return grad[0]

def Hessian_Matrix(L, x):
	# calculate the hessian matrix
    grad = torch.autograd.grad(L(x), x, retain_graph=True, create_graph=True)
    matrix = torch.tensor([])
    for anygrad in grad[0]:
        matrix = torch.cat((matrix, torch.autograd.grad(anygrad, x, retain_graph=True)[0]))
    return matrix.view(x.size()[0], -1)

def Newton(L, x, iters = 10):
    # get the best solution by Newton method
    w = Variable(torch.Tensor(x), requires_grad=True)
    for i in range(iters):
        w.data = w - Hessian_Matrix(L, w).inverse() @ Grad(L, w)   
    return w.detach()
 
def L(x):
    # loss function
    L = -x.t() @ x
    return L
 

x = Variable(torch.Tensor([0.5,5,10]), requires_grad=True)
t = Newton(L, x, iters = 10)
print(t)