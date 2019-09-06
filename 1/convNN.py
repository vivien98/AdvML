import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle
import tensorflow as tf

numClasses = 2
trainSize = numClasses*4000
validSize = numClasses*1000
testSize = numClasses*1000
numClasses = 2
eta = 0.005
sizeLayer1 = 10     
sizeLayer2 = 5	
sizeLayer3 = 10
featLayer1 = 15
featLayer2 = 50
trainingSeries = np.matrix(np.load("../../cifar-10-batches-py/trainingSeries2.npy").astype('float32'))/255
testSeries = np.matrix(np.load("../../cifar-10-batches-py/testSeries2.npy").astype('float32'))/255
trainingLabels = np.load("../../cifar-10-batches-py/trainingLabels2.npy")
testLabels = np.load("../../cifar-10-batches-py/testLabels2.npy")
trainingLabels = trainingLabels[:,0:numClasses]
testLabels = testLabels[:,0:numClasses]



numPixels = 3072
numDots = 32
numChannels = 3



inp = tf.placeholder(tf.float32,shape = [None,numPixels])
inpF = tf.reshape(inp,[-1,numDots,numDots,numChannels])
expected = tf.placeholder(tf.uint8,shape = [None,numClasses])
expectedNum = tf.argmax(expected,axis = 1)

weightsLayer1 = tf.Variable(tf.truncated_normal([sizeLayer1,sizeLayer1,3,featLayer1],stddev = 0.1))
biasesLayer1 = tf.Variable(tf.constant(0.1,shape=[featLayer1]))
layer1 = tf.nn.conv2d(input = inpF,filter = weightsLayer1,strides = [1,1,1,1],padding='SAME')
layer1 += biasesLayer1
layer1 = tf.nn.max_pool(value = layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
layer1 = tf.nn.relu(layer1)

weightsLayer2 = tf.Variable(tf.truncated_normal([sizeLayer2,sizeLayer2,featLayer1,featLayer2],stddev = 0.1))
biasesLayer2 = tf.Variable(tf.constant(0.1,shape=[featLayer2]))
layer2 = tf.nn.conv2d(input = layer1,filter = weightsLayer2,strides = [1,1,1,1],padding='SAME')
layer2 += biasesLayer2
layer2 = tf.nn.max_pool(value = layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
layer2 = tf.nn.relu(layer2)
lshape2 = layer2.get_shape()

layer3in = tf.reshape(layer2,[-1,lshape2[1:4].num_elements()])
weightsLayer3 = tf.Variable(tf.truncated_normal([lshape2[1:4].num_elements(),sizeLayer3],stddev = 0.1))
biasesLayer3 = tf.Variable(tf.constant(0.1,shape=[sizeLayer3]))
layer3out=tf.matmul(layer3in,weightsLayer3) + biasesLayer3
layer3out = tf.nn.relu(layer3out)

weightsLayer4 = tf.Variable(tf.truncated_normal([sizeLayer3,numClasses],stddev=0.1))
biasesLayer4 = tf.Variable(tf.constant(0.6,shape=[numClasses]))
layer4out = tf.matmul(layer3out,weightsLayer4) + biasesLayer4
fin = tf.nn.softmax(layer4out)
finCl = tf.argmax(fin,axis = 1)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4out,labels = expected)
avgLoss = tf.reduce_mean(loss)

opt = tf.train.AdamOptimizer(learning_rate=eta).minimize(avgLoss)

correct = tf.equal(finCl,expectedNum)
acc = tf.reduce_mean(tf.cast(correct,tf.float32))

numWeights = (sizeLayer1*sizeLayer1)*numChannels*featLayer1 + (sizeLayer2*sizeLayer2)*featLayer1*featLayer2 + lshape2[1:4].num_elements()*sizeLayer3 + sizeLayer3*numClasses
print("number of weights : " + str(numWeights))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batchSize = 20
numItr = int(trainSize/batchSize)
valCount = 0
prevacc = 0
while valCount <= 3:
	for i in range(numItr-1):
		xB = trainingSeries[i*batchSize:(i+1)*batchSize,:]
		yB = trainingLabels[i*batchSize:(i+1)*batchSize,:]
		feedDict1 = {inp:xB,expected:yB}
		sess.run(opt,feed_dict = feedDict1)
		if i%100==0:
			xT = trainingSeries[trainSize:trainSize+1000,:]
			yT = trainingLabels[trainSize:trainSize+1000,:]
			feedDict2 = {inp:xT,expected:yT}
			accu = 100*sess.run(acc,feed_dict=feedDict2)
			print("batch "+str(i)+" - "+str(accu))
			if abs(accu-prevacc) < 1:
				valCount+=1
			prevacc = accu
			if valCount>=3:
				break

xTest = testSeries
yTest = testLabels
feedDict3 = {inp:xTest,expected:yTest}
accuF = sess.run(acc,feed_dict=feedDict3)
print("Final accuracy - "+str(accuF))
	




