import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle
import mnist

from rbm1 import RBM

from layers import layer
from layers import relu
from layers import softMax
from layers import crossEntropy
from layers import fcNeuralNet

trainingData, trainingLabels, testData, testLabels = mnist.load()
trainingData = trainingData / 255 > 0.5
testData = testData / 255 > 0.5
trainingData = trainingData.astype(float)
testData = testData.astype(float)

with open("RBM.pkl",'rb') as f:
	rbm1 = pickle.load(f)


trainingSize = 50000	# Number of datapoints per label used in one epoch for training (out of 5000)
validSize = 10000	# Number of datapoints per label used in one epoch for validation (out of 5000)
testSize = 10000		# Number of datapoints per label used in one epoch for testing (out of 1000)
numClasses = 10	# Number of classes to be trained over in MNIST (out of 10)

lay1 = 100			# Number of neurons in input layer
#lay2 = 70			# Number of neurons in first hidden layer
#lay3 = 50			# Number of neurons in second hidden layer

batchSize = 20		# Number of datapoints of per class taken in one step of SGD while training the NN
eta = 0.05			# Learning rate

fcnn = fcNeuralNet(lay1,numClasses)  				#Declared an instance of the fcnn class 				#Results: for 2 classes : (3072,50,10) = 87% accuracy # for 5 classes : (3072,70,50) = 61% accuracy
# numWeights = lay1*lay2+lay2*lay3 +lay3*numClasses			#Calculate and output the number of weights in this network
# print("Number of weights - " + str(numWeights))
trainingData1 = np.zeros((trainingSize + validSize,lay1))
testData1 = np.zeros((testSize,lay1))

for i in range(trainingSize + validSize):
	print("Converting "+str(i))
	trainingData1[i] = rbm1.outputHgivenV(trainingData[i])
	if i < testSize:
		testData1[i] = rbm1.outputHgivenV(testData[i])

# with open('converted.pkl', 'wb') as file1:
#    pickle.dump(conv, file1)

valCount = 0  # This var keeps track of the number of times validation accuracy has been within 1% about its previous value
prevAcc = 0   # Stores accuracy obtained in the previous epoch
epoch  = 0	  #Epoch number

while valCount <= 5:	# Keep training till validation accuracy repeats some number of times
	print("Epoch " + str(epoch))	#print epoch number
	valIndex = trainingSize	# Start validation data from this index in the array trainingSeries
	trainingIndex = 0	#	Start training data from this index in the array trainingSeries
	fcnn.trainOnData(trainingData1,trainingLabels,trainingSize,trainingIndex,eta,batchSize,numClasses)	# Trains the fcnn object on trainingSize number of data points with batchSize as the batch size while doing SGD.
	accuracy = fcnn.validate(validSize,numClasses,valIndex,trainingData1,trainingLabels) #Find validation accuracy after each epoch
	if abs(accuracy - prevAcc) < 1: # increment valCount if previous validation accuracy within 1% of the current value
		valCount += 1 
	else :
		valCount = 0		# Set valCount to 0 if not
	print("validation accuracy " + str(accuracy)) # print validation acuracy for the epoch
	prevAcc = accuracy # previous accuracy updated
	epoch+=1 #increment epoch number


testaccuracy = fcnn.test(testSize,numClasses,testData1,testLabels) # test fcnn on the test dataset after coming out of the validation while loop

print("final accuracy " + str(testaccuracy)) # print test accuracy
 

with open('neuralNet5.pkl', 'wb') as file1:
   pickle.dump(fcnn, file1)