import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl



class layer:

	def __init__(self,inpSize,outSize) :
		
		self.inpSize = inpSize
		self.outSize = outSize
		self.weights = (np.random.rand(outSize,inpSize)-0.5)/10
		self.biases = (np.random.rand(outSize,1)-0.5)
		self.out = np.zeros((outSize,1))
		self.sumDerivWeights = np.zeros(np.shape(self.weights))
		self.sumDerivBiases = np.zeros(np.shape(self.biases))

	def forwardPass(self,inp):
		self.inp = inp 
		out1 = self.weights.dot(inp) + self.biases
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivWeights = forwDeriv.dot(self.inp.T)
		self.derivBiases = forwDeriv
		self.derivIn = self.weights.T.dot(forwDeriv)
		self.sumDerivWeights += self.derivWeights
		self.sumDerivBiases += self.derivBiases

		return self.derivIn

	def resetSums(self):
		self.sumDerivWeights = np.zeros(np.shape(self.weights))
		self.sumDerivBiases = np.zeros(np.shape(self.biases))



class relu:
	def __init__(self,inpSize):
		self.inpSize = inpSize

	def forwardPass(self,inp):
		out1 = np.zeros((self.inpSize,1))
		for i in range(self.inpSize):
			if inp[i,0] > 0:
				out1[i] = inp[i,0]
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivIn = np.zeros((self.inpSize,1))
		for i in range(self.inpSize):
			temp = 0
			if self.out[i] > 0:
				temp = 1
			self.derivIn[i] = forwDeriv[i] * temp
		return self.derivIn
			
class softMax:
	def __init__(self,inpSize):
		self.inpSize = inpSize

	def forwardPass(self,inp):
		out1 = np.exp(inp)
		tot = np.sum(out1)
		out1 = out1/tot 
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivIn = np.zeros((self.inpSize,1))
		s = - self.out.dot(self.out.T)
		self.derivIn = s.dot(forwDeriv) + np.multiply(self.out,forwDeriv)
		return self.derivIn
		
class crossEntropy:
	def __init__(self,inpSize):
		self.inpSize = inpSize

	def forwardPass(self,inp,expected):
		out1 = -expected.T.dot(np.log(inp))
		self.expected = expected
		self.inp = inp
		self.out = out1
		return out1

	def backPass(self,forwDeriv = None):
		self.derivIn = np.zeros((self.inpSize,1))
		for i in range(self.inpSize):
			self.derivIn[i,0] = -self.expected[i,0]/self.inp[i,0]
		return self.derivIn

class fcNeuralNet:
	def __init__(self,*numNeurons):
		self.layers = []
		self.netSize = len(numNeurons)
		for i,obj in enumerate(numNeurons):
			if i < self.netSize-2:
				self.layers.append(layer(numNeurons[i],numNeurons[i+1]))
				self.layers.append(relu(numNeurons[i+1]))
			else:
				if i == self.netSize - 2:
					self.layers.append(layer(numNeurons[i],numNeurons[i+1]))
					self.layers.append(softMax(numNeurons[i+1]))
					self.layers.append(crossEntropy(numNeurons[i+1]))
				else:
					break
		self.layerSize = len(self.layers)
		

	def forwardPassWhileTraining(self,inp,expected):
		temp = inp
		for i,l in enumerate(self.layers):
			if i<self.layerSize-1:
				temp = self.layers[i].forwardPass(temp)
				temp1 = temp
			else:
				temp = self.layers[i].forwardPass(temp,expected)
			

		return temp1

	def forwardPass(self,inp):
		temp = inp
		for i,l in enumerate(self.layers):
			if i<self.layerSize-1:
				temp = self.layers[i].forwardPass(temp)
			else:
				break
			

		return temp
			
	def backPass(self,forwDeriv):
		temp = 1
		for i,l in enumerate(self.layers):
			temp = self.layers[self.layerSize - 1 - i].backPass(temp)

		return temp

	def trainOnBatch(self,batch,eta,batchSize,numClasses):
		for j in range(batchSize):
			for k in range(numClasses):
				mat = np.zeros((numClasses,1))			
				dataPoint = np.matrix(batch[k,j].astype('float64'))
				dataPoint = dataPoint.T / 256
				mat[k,0] = 1
				self.forwardPassWhileTraining(dataPoint,mat)
				self.backPass(1)
		for i,l in enumerate(self.layers):
			if i%2 == 0 and i<self.layerSize-1:
				self.layers[i].weights = self.layers[i].weights - eta*self.layers[i].sumDerivWeights/(batchSize*numClasses)
				self.layers[i].biases = self.layers[i].biases - eta*self.layers[i].sumDerivBiases/(batchSize*numClasses)
				self.layers[i].resetSums()

	def trainOnData(self,trainingData,trainingSize,trainIndex,eta,batchSize,numClasses):
		numItr = int(trainingSize/batchSize)
		for i in range(numItr):
			#print("batchNum " + str(i))
			sample = np.array(trainingData[:,i*batchSize + trainIndex : (i+1)*batchSize + trainIndex])
			self.trainOnBatch(sample,eta,batchSize,numClasses)

	def validate(self,validationSize,numClasses,validIndex,trainingData):
		correct = 0
		for j in range(validationSize):
			for k in range(numClasses):
				dataPoint = np.matrix(trainingData[k,j + validIndex].astype('float64'))
				dataPoint = dataPoint.T/256
				op = self.forwardPass(dataPoint)
				l = np.argmax(op)
				if (k == l) :
					correct += 1

		accuracy = 100*correct/(numClasses*validationSize)
		return accuracy

	def test(self,testSize,numClasses,testData):
		correct = 0
		for j in range(testSize):
			for k in range(numClasses):
				dataPoint = np.matrix(testData[k,j].astype('float64'))
				dataPoint = dataPoint.T/256
				op = self.forwardPass(dataPoint)
				l = np.argmax(op)
				if (k == l) :
					correct += 1

		accuracy = 100*correct/(numClasses*testSize)
		return accuracy





			
			


