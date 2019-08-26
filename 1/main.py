#accuracy = 68% achieved
import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

from layers import layer
from layers import relu
from layers import softMax
from layers import crossEntropy

trainingSeries = np.load("../../cifar-10-batches-py/trainingSeries.npy")
testSeries = np.load("../../cifar-10-batches-py/testSeries.npy")

trainingSize = 3000
validSize = 1000
testSize = 1000
numClasses = 2

batchSize = 10
eta = 0.007

l1 = layer(3072,500)
relu12 = relu(500)
l2 = layer(500,100)
relu23 = relu(100)
l3 = layer(100,2)
softMaxOut = softMax(2)
loss = crossEntropy(2)
weightShape1 = np.shape(l1.weights)
weightShape2 = np.shape(l2.weights)
weightShape3 = np.shape(l3.weights)
biasesShape1 = np.shape(l1.biases)
biasesShape2 = np.shape(l2.biases)
biasesShape3 = np.shape(l3.biases)



numItr = int(trainingSize/batchSize)



for i in range(numItr):
	print("epoch " + str(i))

	sample = np.array(trainingSeries[:,i*batchSize : (i+1)*batchSize])
	weights1 = np.zeros(weightShape1)
	weights2 = np.zeros(weightShape2)
	weights3 = np.zeros(weightShape3)
	biases1 = np.zeros(biasesShape1)
	biases2 = np.zeros(biasesShape2)
	biases3 = np.zeros(biasesShape3)

	for j in range(batchSize):
		

		for k in range(numClasses):
			mat = np.zeros((numClasses,1))			
			dataPoint = np.matrix(sample[k,j].astype('float64'))
			dataPoint = dataPoint.T / 256
			if k ==0:
				mat = np.matrix([[1],[0]])
			else:
				mat = np.matrix([[0],[1]])

			o1=l1.forwardPass(dataPoint)
			o2=relu12.forwardPass(o1)
			o3=l2.forwardPass(o2)
			o4=relu23.forwardPass(o3)
			o5=l3.forwardPass(o4)
			#print(str(o5[0,0]))
			#print(str(o5[1,0]))
			o6=softMaxOut.forwardPass(o5)
			#print(str(o6[0,0]))
			#print(str(o6[1,0]))
			o7=loss.forwardPass(o6,mat)

			b7=loss.backPass()
			b6=softMaxOut.backPass(b7)
			b5=l3.backPass(b6)
			b4=relu23.backPass(b5)
			b3=l2.backPass(b4)
			b2=relu12.backPass(b3)
			b1=l1.backPass(b2)

			weights1 += l1.weights - eta * l1.derivWeights
			weights2 += l2.weights - eta * l2.derivWeights
			weights3 += l3.weights - eta * l3.derivWeights
			biases1 += l1.biases - eta * l1.derivBiases
			biases2 += l2.biases - eta * l2.derivBiases
			biases3 += l3.biases - eta * l3.derivBiases

	l1.weights = weights1/(numClasses*batchSize)
	l2.weights = weights2/(numClasses*batchSize)
	l3.weights = weights3/(numClasses*batchSize)
	l1.biases = biases1/(numClasses*batchSize)
	l2.biases = biases2/(numClasses*batchSize)	
	l3.biases = biases3/(numClasses*batchSize)

	correct = 0

	for i in range(validSize):
		for k in range(numClasses):
			dataPoint = np.matrix(trainingSeries[k,trainingSize + i].astype('float64'))
			dataPoint = dataPoint.T/256
			o1=l1.forwardPass(dataPoint)
			o2=relu12.forwardPass(o1)
			o3=l2.forwardPass(o2)
			o4=relu23.forwardPass(o3)
			o5=l3.forwardPass(o4)
			o6=softMaxOut.forwardPass(o5)
			if (k == 0 and o6[0] >= o6[1]) :
				correct += 1
				#print("this")

			if (k == 1 and o6[0] <= o6[1]):
				correct += 1
				#print("that")
	accuracy = 100*correct/(numClasses*validSize)			

	print("accuracy " + str(accuracy))


		











	
