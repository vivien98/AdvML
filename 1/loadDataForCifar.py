import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import layers

try:
   import cPickle as pickle
except:
   import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

numd = 50000
numt = 10000

op1 = unpickle("../../cifar-10-batches-py/data_batch_1")
op2 = unpickle("../../cifar-10-batches-py/data_batch_2")
op3 = unpickle("../../cifar-10-batches-py/data_batch_3")
op4 = unpickle("../../cifar-10-batches-py/data_batch_4")
op5 = unpickle("../../cifar-10-batches-py/data_batch_5")
test = unpickle("../../cifar-10-batches-py/test_batch")

dataSeries1 = op1[b'data']
dataExpected1 = op1[b'labels']
dataSeries2 = op2[b'data']
dataExpected2 = op2[b'labels']
dataSeries3 = op3[b'data']
dataExpected3 = op3[b'labels']
dataSeries4 = op4[b'data']
dataExpected4 = op4[b'labels']
dataSeries5 = op5[b'data']
dataExpected5 = op5[b'labels']
testSeries = test[b'data']
testExpected = test[b'labels']

dataSeries = np.append(dataSeries1,np.append(dataSeries2,np.append(dataSeries3,np.append(dataSeries4,dataSeries5,axis = 0),axis = 0),axis = 0),axis = 0)
dataExpected = np.append(dataExpected1,np.append(dataExpected2,np.append(dataExpected3,np.append(dataExpected4,dataExpected5,axis = 0),axis = 0),axis = 0),axis = 0)

trainingSeries = np.zeros((10,5000,3072))
count = np.zeros(10)

for i in range(numd):
	trainingSeries[int(dataExpected[i]),int(count[int(dataExpected[i])])] = dataSeries[i]
	count[dataExpected[i]] += 1

evalSeries = np.zeros((10,1000,3072))
count = np.zeros(10)

for i in range(numt):
	evalSeries[int(testExpected[i]),int(count[int(testExpected[i])])] = testSeries[i]
	count[testExpected[i]] += 1

np.save('trainingSeries',trainingSeries)
np.save('testSeries',evalSeries)


# [b'airplane',
#  b'automobile',
#  b'bird',
#  b'cat',
#  b'deer',
#  b'dog',
#  b'frog',
#  b'horse',
#  b'ship',
#  b'truck']
