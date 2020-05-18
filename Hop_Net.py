# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:20:49 2020

@author: utenti
"""

import numpy as np
from sklearn.datasets import load_digits

def matrix2array(matrix):
    return matrix.flatten()

def createWeightMatrix(imageAsArray):
    n = np.shape(imageAsArray)[0]
    weights = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            if i != j:
                weights[i,j] = imageAsArray[i] * imageAsArray[j]
                weights[j,i] = weights[i,j]
    return weights

def main():
    x = np.ones((2,2))
    x = x + x
    a = matrix2array(x)
    y = np.ones((2,2))
    a2 = matrix2array(y)
    w = createWeightMatrix(a)
    w += createWeightMatrix(a2)
    w /= 2
    print(w)

main()
    