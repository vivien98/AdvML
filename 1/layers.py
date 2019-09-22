import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

#-----------------------------------------Linear layer______________________________________________________________________#

class layer:		# Defines a linear layer in the NN

	def __init__(self,inpSize,outSize) : #initialize the layer
		
		self.inpSize = inpSize # number of input neurons
		self.outSize = outSize # number of output neurons
		self.weights = (np.random.rand(outSize,inpSize)-0.5)/10 # initialize weights randomly between -0.1 and 0.1
		self.biases = (np.random.rand(outSize,1)-0.5) # initialize biases randomly between -1 and 1
		self.out = np.zeros((outSize,1)) # output initialized
		self.sumDerivWeights = np.zeros(np.shape(self.weights))	# variable to store the sum of derivatives of loss with respect to weights for a particular batch in SGD
		self.sumDerivBiases = np.zeros(np.shape(self.biases))	# variable to store the sum of derivatives of loss with respect to biases for a particular batch in SGD

	def forwardPass(self,inp): 						# return output as Wx + b
		self.inp = inp 								# store input for use while backprop
		out1 = self.weights.dot(inp) + self.biases 	# output = Weights.Inputs + biases
		self.out = out1								# store output for use while backprop
		return out1

	def backPass(self,forwDeriv): # return derivatives wrt inputs and store the derivatives wrt weights,biases. Here forwDeriv is the derivative of the loss wrt the inputs of the layer after this one
		self.derivWeights = forwDeriv.dot(self.inp.T)	# derivatives wrt weights stored
		self.derivBiases = forwDeriv					# derivatives wrt biases stored
		self.derivIn = self.weights.T.dot(forwDeriv)	# derivatives wrt inputs of this layer
		self.sumDerivWeights += self.derivWeights       # store the sum of derivatives wrt weights of this batch for averaging later for SGD
		self.sumDerivBiases += self.derivBiases         # store the sum of derivatives wrt biases of this batch for averaging later for SGD

		return self.derivIn

	def resetSums(self): # reset the sums of all derivatives in a batch after finishing training of the batch
		self.sumDerivWeights = np.zeros(np.shape(self.weights))
		self.sumDerivBiases = np.zeros(np.shape(self.biases))


#-----------------------------------------Relu layer______________________________________________________________________#


class relu:			# ReLU layer class
	def __init__(self,inpSize):
		self.inpSize = inpSize # initialize the size of the input

	def forwardPass(self,inp): # returns the output of the layer given the input inp
		out1 = np.zeros((self.inpSize,1)) # initialize output size = input size
		for i in range(self.inpSize): # for each input 
			if inp[i,0] > 0: # if input is positive
				out1[i] = inp[i,0] # then output = input
		self.out = out1 # store output for use while backpropogating
		return out1

	def backPass(self,forwDeriv): # given the derivative wrt the output (i.e. the input of the next layer), return the derivative wrt input of this layer
		self.derivIn = np.zeros((self.inpSize,1)) # derivative wrt input initialised
		for i in range(self.inpSize): 
			temp = 0
			if self.out[i] > 0: # derivative of relu = 1 only when value of relu > 0
				temp = 1
			self.derivIn[i] = forwDeriv[i] * temp # chain rule
		return self.derivIn


#-----------------------------------------Softmax layer______________________________________________________________________#
			
class softMax: 			#Softmax layer class
	def __init__(self,inpSize): # Initialize the size of the input
		self.inpSize = inpSize

	def forwardPass(self,inp): # return output for a given input to the layer
		out1 = np.exp(inp) # output computed
		tot = np.sum(out1) # normalizing factor computed
		out1 = out1/tot 	# normalized output
		self.out = out1 # output stored for use while backprop
		return out1	

	def backPass(self,forwDeriv): # return derivative of loss wrt input given the derivatives wrt the output
		self.derivIn = np.zeros((self.inpSize,1)) # initialize derivatives
		s = - self.out.dot(self.out.T)            
		self.derivIn = s.dot(forwDeriv) + np.multiply(self.out,forwDeriv) # Derivatives calculated wrt inputs  (formula for derivative of softmax written in terms of the outputs of the softmax function)
		return self.derivIn

#-----------------------------------------Loss layer______________________________________________________________________#

class crossEntropy: # cross entropy layer class
	def __init__(self,inpSize): # Initialize the size of the input
		self.inpSize = inpSize

	def forwardPass(self,inp,expected): # return the loss given input and the expected output 
		out1 = -expected.T.dot(np.log(inp)) # cross entropy calculated 
		self.expected = expected # store expected output, input and output for use while backprop
		self.inp = inp
		self.out = out1
		return out1

	def backPass(self,forwDeriv = None): # return the derivative of loss wrt input of this layer
		self.derivIn = np.zeros((self.inpSize,1))
		for i in range(self.inpSize): # for all inputs
			self.derivIn[i,0] = -self.expected[i,0]/self.inp[i,0] # derivative calculated for a particular input
		return self.derivIn


#----------------------------------------------------------------- Main Class ____________________________________________________________________________ #

class fcNeuralNet:  # class for the fully connected neural net

	def __init__(self,*numNeurons):
		self.layers = [] # list of layers of this neural net
		self.netSize = len(numNeurons) # number of layers (where one layer = linear + relu/softmax)
		for i,obj in enumerate(numNeurons):
			if i < self.netSize-2: # if we are not constructing the last layer
				self.layers.append(layer(numNeurons[i],numNeurons[i+1])) # append a linear layer to the list
				self.layers.append(relu(numNeurons[i+1])) # append a relu layer to the list
			else:
				if i == self.netSize - 2:# if last layer is being constructed
					self.layers.append(layer(numNeurons[i],numNeurons[i+1])) # append a linera layer
					self.layers.append(softMax(numNeurons[i+1])) # append a softmax layer
					self.layers.append(crossEntropy(numNeurons[i+1])) # append a cross entropy loss layer
				else:
					break
		self.layerSize = len(self.layers) # number of layers (where one layer = one instance of any of the classes defined above)
		

	def forwardPassWhileTraining(self,inp,expected): # do a forwrd pass of the whole network given a data point inp, also calculate the loss at the end
		temp = inp
		for i,l in enumerate(self.layers):
			if i<self.layerSize-1:
				temp = self.layers[i].forwardPass(temp) # do forward pass on all layers
				temp1 = temp
			else:
				temp = self.layers[i].forwardPass(temp,expected) # for the last layers calculate the loss wrt expected output too
			

		return temp1 # return softmax output

	def forwardPass(self,inp):# do a forwrd pass of the whole network given a data point inp,without calculating loss
		temp = inp
		for i,l in enumerate(self.layers):
			if i<self.layerSize-1:
				temp = self.layers[i].forwardPass(temp) # do forward pass on all layers
			else:
				break
			

		return temp #return the softmax output
			
	def backPass(self,forwDeriv): # backpropagate for a particular data point through all layers of the fcnn
		temp = 1
		for i,l in enumerate(self.layers):
			temp = self.layers[self.layerSize - 1 - i].backPass(temp) # apply the backpass function on all layers, pass the output of a layer ahead in the list to the one behind

		return temp

	def trainOnBatch(self,batch,eta,batchSize,numClasses):  # train the weights and biases of the fcnn for a given batch
		for j in range(batchSize): # for each datapoint per label
			for k in range(numClasses): # for each label
				mat = np.zeros((numClasses,1))			
				dataPoint = np.matrix(batch[k,j].astype('float64'))
				dataPoint = dataPoint.T / 256				# make datapoint pixel values map to [0,1]
				mat[k,0] = 1   # make expected output as one hot encoded
				self.forwardPassWhileTraining(dataPoint,mat) # do a forward pass on the network
				self.backPass(1) # backpropogate immediately after the fwd pass 
		for i,l in enumerate(self.layers):  # for each layer l
			if i%2 == 0 and i<self.layerSize-1: # if the layer is linear 
				self.layers[i].weights = self.layers[i].weights - eta*self.layers[i].sumDerivWeights/(batchSize*numClasses) #update its weights by subtracting the learning rate multiplied by averaged derivative over the batch from each weight 
				self.layers[i].biases = self.layers[i].biases - eta*self.layers[i].sumDerivBiases/(batchSize*numClasses) #update its biases by subtracting the learning rate multiplied by averaged derivative over the batch from each bias
				self.layers[i].resetSums() # reset the accumulated sums of derivatives in each layer before training the next batch

	def trainOnData(self,trainingData,trainingSize,trainIndex,eta,batchSize,numClasses): # train the fcnn on a given dataset
		numItr = int(trainingSize/batchSize) # number of batch training iterations
		for i in range(numItr): # for all batches
			sample = np.array(trainingData[:,i*batchSize + trainIndex : (i+1)*batchSize + trainIndex]) # batch data collected from training data
			self.trainOnBatch(sample,eta,batchSize,numClasses) # train the fcnn on the batch data

	def validate(self,validationSize,numClasses,validIndex,trainingData): # test the fcnn on validation data
		correct = 0
		for j in range(validationSize):	# for each datapoint per label
			for k in range(numClasses): # for each label
				dataPoint = np.matrix(trainingData[k,j + validIndex].astype('float64'))
				dataPoint = dataPoint.T/256 # make datapoint pixel values map to [0,1]
				op = self.forwardPass(dataPoint) # calculate output on datapoint
				l = np.argmax(op) # predicted class = argmax(softmmax output of fcnn)
				if (k == l) : # compare predicted class with expected class
					correct += 1

		accuracy = 100*correct/(numClasses*validationSize) # calculate accuracy
		return accuracy

	def test(self,testSize,numClasses,testData): # test the fcnn on test data (same format as the function validate())
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





			
			


