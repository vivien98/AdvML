import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle

from layers import layer
from layers import relu
from layers import softMax
from layers import crossEntropy
from layers import fcNeuralNet

trainingSeries = np.load("../../cifar-10-batches-py/trainingSeries.npy")
testSeries = np.load("../../cifar-10-batches-py/testSeries.npy")

trainingSize = 4000
validSize = 1000
testSize = 1000
numClasses = 5

batchSize = 5
eta = 0.01

fcnn = fcNeuralNet(3072,70,50,numClasses)  # for 2 classes : (3072,50,10) = 87% accuracy # for 5 classes :


numItr = int(trainingSize/batchSize)
valCount = 0
prevAcc = 0
epoch  = 0

while valCount <= 5:
	print("Epoch " + str(epoch))
	valIndex = trainingSize
	trainingIndex = 0
	fcnn.trainOnData(trainingSeries,trainingSize,trainingIndex,eta,batchSize,numClasses)
	accuracy = fcnn.validate(validSize,numClasses,valIndex,trainingSeries)
	if abs(accuracy - prevAcc) < 1:
		valCount += 1 
	else :
		valCount = 0		
	print("validation accuracy " + str(accuracy))
	prevAcc = accuracy
	epoch+=1


testaccuracy = fcnn.test(testSize,numClasses,testSeries)

print("final accuracy " + str(testaccuracy))
 

with open('neuralNet5.pkl', 'wb') as file1:
   pickle.dump(fcnn, file1)