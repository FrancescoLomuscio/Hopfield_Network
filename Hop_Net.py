# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:20:49 2020

@author: utenti
"""
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data
from skimage.transform import resize


def matrix2array(matrix):
    return np.asarray(matrix).flatten()

def createWeightMatrix(imagesAsArray):
    n = np.shape(imagesAsArray[0])[0]
    weights = np.zeros((n,n))
    for image in imagesAsArray:
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
    fundamentalMemories = [data.coins(), data.camera()]
    fundamentalMemories_ = []
    for image in fundamentalMemories:
        basewidth = 50
        image = resize(image, (basewidth,basewidth))
        #print(image)
        image = binarizeImage(np.asarray(image))
        fundamentalMemories_.append(image)
    return fundamentalMemories_

def binarizeImage(image):
    _image = np.zeros((image.shape[0],image.shape[1]))
    for i in range(_image.shape[0]):
        for j in range(_image.shape[1]):
            if image[i,j] >= 0.5:
                _image[i,j] = 1
            else:
                _image[i,j] = -1

    return _image
def update(weights, y, iterations = 10000):
    for s in range(iterations):
        n = np.shape(y)[0]
        i = np.random.randint(0,n-1)
        vi = np.dot(weights[i,:],y)
        yi = activation(vi)
        y[i] = yi
    return y

def activation(vi):
    if vi > 0:
        return 1
    elif vi < 0:
        return -1
    else:
        return vi
        
def main():
    memories = getFundamentalMemories()

    for im in memories:
        plt.imshow(im)
        plt.show()
       
    m = []
    for el in memories:
        m.append(matrix2array(el))
        
    w = createWeightMatrix(m)
        
    tests = getFundamentalMemories()
    for test in tests:
        test[10:30,10:20] = -1
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(test)
        #axarr[0].title("Immagine n."+str(i))
        axarr[1].imshow(np.reshape(update(w,matrix2array(test)),(50,50)))
        #axarr[1].title("Ricostruzione immagine n."+str(i))
    
        plt.show()
#"""
    

main()
    