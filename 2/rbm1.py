import numpy as np
import math
import pickle
import matplotlib.pyplot as plt



class RBM:

	def __init__(self,visibleSize,hiddenSize):
		self.weights = (np.random.rand(visibleSize,hiddenSize)-0.5)/100
		self.visBiases = (np.random.rand(1,visibleSize)-0.5)/100
		self.hidBiases = (np.random.rand(1,hiddenSize)-0.5)/100
		self.visibleSize = visibleSize
		self.hiddenSize = hiddenSize

	def sampleHgivenV(self,vis):
		probVecH = 1/(1+np.exp(-(np.matmul(vis,self.weights)+self.hidBiases)))
		sampledH = np.random.binomial(1,probVecH)[0,:]
		sampledH = sampledH.astype(float)
		return sampledH

	def outputHgivenV(self,vis):
		probVecH = 1/(1+np.exp(-(np.matmul(vis,self.weights)+self.hidBiases)))
		return probVecH

	def sampleVgivenH(self,hid):
		probVecV = 1/(1+np.exp(-(np.matmul(hid,self.weights.T)+self.visBiases)))
		sampledV = np.random.binomial(1,probVecV)[0,:]
		sampledV = sampledV.astype(float)
		return sampledV

	def GibbSample(self,k,v0):
		hArray = np.zeros((k,self.hiddenSize))
		vArray = np.zeros((k,self.visibleSize))
		vArray[0,:] = v0
		vCurr = v0
		for i in range(k):
			hCurr = self.sampleHgivenV(vCurr)
			vNext = self.sampleVgivenH(hCurr)
			hArray[i,:] = hCurr
			vCurr = vNext
			if i != k-1:
				vArray[i+1,:] = vCurr

		return vArray,hArray

	def addInGrad(self,k,v0,vk,hArray):
		probVecH0 = hArray[0,:].reshape(1,self.hiddenSize) #1/(1+np.exp(-(np.matmul(v0,self.weights)+self.hidBiases)))
		probVecHk = hArray[k-1,:].reshape(1,self.hiddenSize)#1/(1+np.exp(-(np.matmul(vk,self.weights)+self.hidBiases)))
		gradWeight = np.matmul(v0.T,probVecH0) - np.matmul(vk.T,probVecHk) 
		gradVisBias = (v0 - vk)
		gradHidBias = probVecH0 - probVecHk
		return gradWeight,gradVisBias,gradHidBias


	def trainOnBatchCD(self,k,eta,lam,alpha,trainingBatch,batchSize):
		gradWeightSum = np.zeros((self.visibleSize,self.hiddenSize))
		gradVisBiasSum = np.zeros((1,self.visibleSize))
		gradHidBiasSum = np.zeros((1,self.hiddenSize))

		for i in range(batchSize):
			visibleData = trainingBatch[i]
			vArray,hArray = self.GibbSample(k,visibleData)
			v0 = visibleData.reshape(1,784)
			vk = vArray[k-1,:].reshape(1,784)			
			gradWeight,gradVisBias,gradHidBias = self.addInGrad(k,v0,vk,hArray)
			gradWeightSum = gradWeightSum + gradWeight
			gradVisBiasSum = gradVisBiasSum + gradVisBias
			gradHidBiasSum = gradHidBiasSum + gradHidBias

		gradWeightSum = gradWeightSum / batchSize
		gradVisBiasSum = gradVisBiasSum / batchSize
		gradHidBiasSum = gradHidBiasSum / batchSize
		
		self.velWeights = alpha*self.velWeights + eta*(gradWeightSum - lam*self.weights)
		self.velVisBias = alpha*self.velVisBias + eta*(gradVisBiasSum - lam*self.visBiases)
		self.velHidBias = alpha*self.velHidBias + eta*(gradHidBiasSum - lam*self.hidBiases)
		
		self.weights = self.weights + self.velWeights
		self.visBiases = self.visBiases + self.velVisBias
		self.hidBiases = self.hidBiases + self.velHidBias

	def testOnSample(self,k,testData,testSize,showH):
		sqroot = int(math.sqrt(testSize)) + 1
		for l in range(testSize):
			temp,dump = self.GibbSample(k,testData[l].T)
			for i in range(k):
				ax = plt.subplot(testSize,k,k*l+i+1)
				ax.set_xticks([])
				ax.set_yticks([])
				plt.imshow(temp[i,:].reshape(1,784).reshape(28,28),cmap="gray")
		plt.show()
		if showH == 1:
			for l in range(testSize):
				temp,dump = self.GibbSample(k,testData[l].T)
				for i in range(k):
					ax1 = plt.subplot(testSize,k,k*l+i+1)
					ax1.set_xticks([])
					ax1.set_yticks([])
					plt.imshow(dump[i,:].reshape(1,100).reshape(10,10),cmap="gray")
			plt.show()

	def showFilters(self,k):
		sqroot = int(math.sqrt(self.hiddenSize)) + 1
		for l in range(self.hiddenSize):
			ax = plt.subplot(sqroot,sqroot,l+1)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(self.weights[:,l].reshape(28,28),cmap="gray")
		plt.show()


	def trainUsingCD(self,k,eta,lam,alpha,trainingData,trainingSize,batchSize,numEpochs,testData,testSize):
		numItr = int(trainingSize/batchSize)
		#self.testOnSample(k,testData,testSize,1)
		for j in range(numEpochs):
			print("Epoch number : " + str(j))
			self.velWeights = np.zeros((self.visibleSize,self.hiddenSize))
			self.velVisBias = np.zeros((1,self.visibleSize))
			self.velHidBias = np.zeros((1,self.hiddenSize))
			for i in range(numItr):
				print("Batch Number : " + str(i) +" out of " + str(numItr))
				trainingBatch = trainingData[i*batchSize:(i+1)*batchSize,:].reshape(batchSize,784)
				self.trainOnBatchCD(k,eta,lam,alpha,trainingBatch,batchSize)
		self.testOnSample(k,testData,testSize,0)
		self.showFilters(k)