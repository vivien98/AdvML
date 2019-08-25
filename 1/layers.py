import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl

class layer:

	def __init__(self,inpSize,weights,biases,outSize) :
		
		self.inpSize = inpSize
		self.outSize = outSize
		self.weights = weights
		self.biases = biases
		self.out = np.zeros((outSize,1))

	def forwardPass(self,inp,weights = self.weights,biases = self.biases):
		self.weights = weights
		self.biases = biases
		self.inp = inp 
		out1 = self.weights * inp + self.biases
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivWeights = forwDeriv * self.inp.T
		self.derivBiases = forwDeriv
		self.derivIn = self.weights.T * forwDeriv

		return self.derivIn



class relu:
	def __init__(self,inpSize):
		self.inpSize = inpSize

	def forwardPass(self,inp):
		out1 = np.zeros((len(inp),1))
		for i in range(len(inp)):
			out1[i][0] = max(0,inp[i]) 
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivIn = np.zeros((self.inpSize,1))
		for i in range(inpSize):
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
		tot = np.sum(out)
		out1 = out/tot 
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivIn = np.zeros((self.inpSize,1))
		s = - self.out * self.out.T
		self.derivIn = s * forwDeriv + np.multiply(self.out,forwDeriv)
		return self.derivIn
		
class crossEntropy:
	def __init__(self,inpSize):
		self.inpSize = inpSize

	def forwardPass(self,inp,expected):
		out1 = -(expected.T)*(np.log(inp))
		self.expected = expected
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivIn = np.zeros((self.inpSize,1))
		for i in range(self.inpSize):
			self.derivIn[i] = -self.expected[i][0]/self.out[i][0]
		return self.derivIn
			


