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

	def forwardPass(self,inp):
		self.inp = inp 
		out1 = self.weights.dot(inp) + self.biases
		self.out = out1
		return out1

	def backPass(self,forwDeriv):
		self.derivWeights = forwDeriv.dot(self.inp.T)
		self.derivBiases = forwDeriv
		self.derivIn = self.weights.T.dot(forwDeriv)

		return self.derivIn



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
			


