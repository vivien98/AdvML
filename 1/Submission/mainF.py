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

trainingSeries = np.load("trainingSeries.npy")	# All training data is loaded into this array (format is : trainingSeries[label number(0 to 9) , image number (0 to 4999), pixel number (0 to 3071)])
testSeries = np.load("testSeries.npy")	# All testing data is loaded into this array (same format as training data)

trainingSize = 4000	# Number of datapoints per label used in one epoch for training (out of 5000)
validSize = 1000	# Number of datapoints per label used in one epoch for validation (out of 5000)
testSize = 1000		# Number of datapoints per label used in one epoch for testing (out of 1000)
numClasses = 5		# Number of classes to be trained over in CIFAR (out of 10)

lay1 = 3072			# Number of neurons in input layer
lay2 = 70			# Number of neurons in first hidden layer
lay3 = 50			# Number of neurons in second hidden layer

batchSize = 5		# Number of datapoints of per class taken in one step of SGD while training the NN
eta = 0.01			# Learning rate

fcnn = fcNeuralNet(lay1,lay2,lay3,numClasses)  				#Declared an instance of the fcnn class 				#Results: for 2 classes : (3072,50,10) = 87% accuracy # for 5 classes : (3072,70,50) = 61% accuracy
numWeights = lay1*lay2+lay2*lay3 +lay3*numClasses			#Calculate and output the number of weights in this network
print("Number of weights - " + str(numWeights))


valCount = 0  # This var keeps track of the number of times validation accuracy has been within 1% about its previous value
prevAcc = 0   # Stores accuracy obtained in the previous epoch
epoch  = 0	  #Epoch number

while valCount <= 5:	# Keep training till validation accuracy repeats some number of times
	print("Epoch " + str(epoch))	#print epoch number
	valIndex = trainingSize	# Start validation data from this index in the array trainingSeries
	trainingIndex = 0	#	Start training data from this index in the array trainingSeries
	fcnn.trainOnData(trainingSeries,trainingSize,trainingIndex,eta,batchSize,numClasses)	# Trains the fcnn object on trainingSize number of data points with batchSize as the batch size while doing SGD.
	accuracy = fcnn.validate(validSize,numClasses,valIndex,trainingSeries) #Find validation accuracy after each epoch
	if abs(accuracy - prevAcc) < 1: # increment valCount if previous validation accuracy within 1% of the current value
		valCount += 1 
	else :
		valCount = 0		# Set valCount to 0 if not
	print("validation accuracy " + str(accuracy)) # print validation acuracy for the epoch
	prevAcc = accuracy # previous accuracy updated
	epoch+=1 #increment epoch number


testaccuracy = fcnn.test(testSize,numClasses,testSeries) # test fcnn on the test dataset after coming out of the validation while loop

print("final accuracy " + str(testaccuracy)) # print test accuracy
 

with open('neuralNet5.pkl', 'wb') as file1:
   pickle.dump(fcnn, file1)