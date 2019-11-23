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

op1 = unpickle("data_batch_1")
op2 = unpickle("data_batch_2")
op3 = unpickle("data_batch_3")
op4 = unpickle("data_batch_4")
op5 = unpickle("data_batch_5")
test = unpickle("test_batch")

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
trainingSeries2 = np.zeros((5000*2,3072))
trainingLabels2 = np.zeros((5000*2,10))
testSeries2 = np.zeros((1000*2,3072))
testLabels2 = np.zeros((1000*2,10))
trainingSeries5 = np.zeros((5000*5,3072))
trainingLabels5 = np.zeros((5000*5,10))
testSeries5 = np.zeros((1000*5,3072))
testLabels5 = np.zeros((1000*5,10))
count = np.zeros(10)
cnt2 =0
cnt5 = 0

for i in range(numd):
	trainingSeries[int(dataExpected[i]),int(count[int(dataExpected[i])])] = dataSeries[i]
	count[dataExpected[i]] += 1
	#-----#
	if dataExpected[i] >=0 and dataExpected[i] <=1:
		trainingLabels2[cnt2,int(dataExpected[i])] = 1 
		trainingSeries2[cnt2]=dataSeries[i]
		cnt2+=1

	if dataExpected[i] >=0 and dataExpected[i] <=4:
		trainingLabels5[cnt5,int(dataExpected[i])] = 1 
		trainingSeries5[cnt5]=dataSeries[i]
		cnt5+=1



evalSeries = np.zeros((10,1000,3072))
count = np.zeros(10)
cnt2=0
cnt5=0
for i in range(numt):
	evalSeries[int(testExpected[i]),int(count[int(testExpected[i])])] = testSeries[i]
	count[testExpected[i]] += 1
	# #-----#
	if testExpected[i] >=0 and testExpected[i] <=1:
		testLabels2[cnt2,int(testExpected[i])] = 1 
		testSeries2[cnt2]=testSeries[i]
		cnt2+=1

	if testExpected[i] >=0 and testExpected[i] <=4:
		testLabels5[cnt5,int(testExpected[i])] = 1 
		testSeries5[cnt5]=testSeries[i]
		cnt5+=1


np.save('trainingSeries',trainingSeries)
np.save('testSeries',evalSeries)
np.save('trainingSeries2',trainingSeries2)
np.save('testSeries2',testSeries2)
np.save('trainingLabels2',trainingLabels2)
np.save('testLabels2',testLabels2)
np.save('trainingSeries5',trainingSeries5)
np.save('testSeries5',testSeries5)
np.save('trainingLabels5',trainingLabels5)
np.save('testLabels5',testLabels5)




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
