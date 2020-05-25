# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:20:49 2020

@author: utenti
"""

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from PIL import Image


def matrix2array(matrix):
    return matrix.flatten()

def createWeightMatrix(imagesAsArray):
    n = np.shape(imagesAsArray[0])[0]
    weights = np.zeros((n,n))
    for image in imagesAsArray:
        image = matrix2array(image)
        t = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                if i != j:
                    t[i,j] = image[i] * image[j]
                    t[j,i] = t[i,j]
        weights += t
    return weights/len(imagesAsArray)

def gray2binary(image):
    gray = np.dot(image,[0.2989, 0.5870, 0.1140])
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j] < 128:
                gray[i,j] = -1
            else:
                gray[i,j] = 1

    return gray

def getFundamentalMemories():
    digits = load_digits()
    fm = [digits.images[0],digits.images[7]]
    for image in fm:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j] == 0:
                    image[i,j] = -1
                else:
                    image[i,j] = 1
    return fm
        
def main():
    memories = getFundamentalMemories()
    w = createWeightMatrix(memories)
    print(w)
    

main()
    