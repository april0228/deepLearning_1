# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:35:02 2021
@author: HASEE
deep learning for cat forsee 1
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

def loadDataset():
    
    loc = "./datasets/"
    
    trainSet = h5.File(loc + str("train_catvnoncat.h5"), "r")
    trainSetX = np.array(trainSet["train_set_x"][:])
    trainSetY = np.array(trainSet["train_set_y"][:])
    trainSetX = (trainSetX.reshape(trainSetX.shape[0],-1))/ 255.0
    trainSetY = trainSetY.reshape((1,trainSetY.shape[0])).T
    
    testSet = h5.File(loc + str("test_catvnoncat.h5"), "r")
    testSetX = np.array(testSet["test_set_x"][:])
    testSetY = np.array(testSet["test_set_y"][:])
    testSetX = (testSetX.reshape(testSetX.shape[0],-1)) / 255.0
    testSetY = testSetY.reshape((1,testSetY.shape[0])).T
    
    
    return trainSetX,trainSetY,testSetX,testSetY
def sigmoid(w,x,b):
    
    z = np.dot(x,w) + b
    a = 1/(1 + np.exp(-z))
    return a
    
def parameterInit() :
    w = np.zeros((12288,1))
    b = 0
    time = 2000
    r = 0.005
    return w,b,time,r
    
def main():
    
    trainSetX,trainSetY,testSetX,testSetY = loadDataset()
    w,b,time,r = parameterInit()
    """
    print("trainsetx_shape :" , trainSetX.shape)
    print("trainsety_shape :" , trainSetY.shape)
    print("testsetx_shape :" , testSetX.shape)
    print("testsety_shape :" , testSetY.shape)
    trainsetx_shape : (209, 12288)
    trainsety_shape : (209, 1)
    testsetx_shape : (50, 12288)
    testsety_shape : (50, 1)
    w: (12288,1)
    """
    t = 0
    n = 209
    while(t <= time):
        A = sigmoid(w,trainSetX,b) # A (209,1)
        loss = -np.sum(trainSetY*(np.log(A)) + (1-trainSetY)*(np.log(1-A))) / n  
        dz = A - trainSetY  # dz (209,1)
        dw = (np.dot(trainSetX.T,dz)) / n
        db = np.sum(dz) / n
        w = w - r*dw
        b = b - r*db
        if(t%50 == 0 and 0) :
            print(t,"th,loss = ",loss)
        t = t + 1
    
    #计算训练集准确率
    A = sigmoid(w,trainSetX,b)
    for i in range( A.shape[0] ) :
        if(A[i] > 0.5):
            A[i] = 1
        else:
            A[i] = 0
    print("训练集准确率：",np.sum(A==trainSetY)/209)
    
    #计算测试集准确率
    A = sigmoid(w,testSetX,b)
    for i in range( A.shape[0] ) :
        if(A[i] > 0.5):
            A[i] = 1
        else:
            A[i] = 0
    print("测试集准确率：",np.sum(A==testSetY)/50)
    return 0
main()



