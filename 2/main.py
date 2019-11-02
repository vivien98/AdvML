import numpy as np
import math
import pickle
import mnist
import matplotlib.pyplot as plt
import pickle

from rbm1 import RBM

trainingData, trainingLabels, testData, testLabels = mnist.load()

trainingData = trainingData / 255 > 0.5
testData = testData / 255 > 0.5

trainingData = trainingData.astype(float)
testData = testData.astype(float)

eta = 0.01
lam = 0.01
alpha = 0.9
k = 20

visibleSize = 784
hiddenSize = 100

trainingSize = 60000
batchSize = 20
numEpochs = 5

RBM1 = RBM(visibleSize,hiddenSize)

RBM1.trainUsingCD(k,eta,lam,alpha,trainingData,trainingSize,batchSize,numEpochs,testData[0:10,:],10)

with open('RBM.pkl', 'wb') as file1:
   pickle.dump(RBM1, file1)