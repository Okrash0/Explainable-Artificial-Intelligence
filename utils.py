# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:53:56 2023

@author: Baumann
"""

import numpy
import random
import copy
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

# --------------------------------------
# Load parameters
# --------------------------------------

def loadparams():
    W = [numpy.loadtxt('params/l%d-W.txt'%l) for l in range(1,4)]
    B = [numpy.loadtxt('params/l%d-B.txt'%l) for l in range(1,4)]
    return W,B

# --------------------------------------
# Load data
# --------------------------------------

def loaddata():
    X = numpy.loadtxt('data/X.txt')
    T = numpy.loadtxt('data/T.txt')
    return X,T

# --------------------------------------
# Visualizing data
# --------------------------------------

def heatmap(R,sx,sy):

    b = 10*((numpy.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(numpy.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    plt.show()

def digit(X,sx,sy):

    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(X,interpolation='nearest',cmap='gray')
    plt.show()

def image(X,sx,sy):

    plt.figure(figsize=(sx,sy))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.axis('off')
    plt.imshow(X,interpolation='nearest')
    plt.show()

# --------------------------------------------------------------
# Clone a layer and pass its parameters through the function g
# --------------------------------------------------------------

def newlayer(layer,g):

    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer

# --------------------------------------------------------------
# convert VGG classifier's dense layers to convolutional layers
# --------------------------------------------------------------

def toconv(layers):

    newlayers = []

    for i,layer in enumerate(layers):
        print(layer)
        if isinstance(layer,nn.Linear):

            newlayer = None

            if i == 0:
                m,n = 512,layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,7)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,7,7))

            else:
                print(layer.weight.shape)
                m,n = layer.weight.shape[1],layer.weight.shape[0]
                newlayer = nn.Conv2d(m,n,1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n,m,1,1))

            newlayer.bias = nn.Parameter(layer.bias)

            newlayers += [newlayer]

        else:
            newlayers += [layer]

    return newlayers

# --------------------------------------------------------------
# my own stuff...
# --------------------------------------------------------------

class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output): 
        self.features = ((output.cpu()).data).numpy()
    
    def remove(self): 
        self.hook.remove()
  
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - numpy.min(cam)
    cam_img = cam / numpy.max(cam)
    return [cam_img]

def noise(arrn, kind='gauss'):
    arr = copy.deepcopy(arrn)
    if kind == 'gauss':
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                for k in range(len(arr[i, j])):
                    arr[i, j, k] += random.gauss(0, 1)
    elif kind == 'sp':
        for i in range(len(arr)):
            for j in range(len(arr[i])):#
                for k in range(len(arr[i, j])):
                    
                    temp = random.random()    
                    if temp < 0.1:
                        arr[i, j, k] = -2.5
                    elif temp > 0.9:
                        arr[i, j, k] = 2.5
    elif kind == 'flip':
        arr = arr.flip(1)
    return arr


