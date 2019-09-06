import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle
import tensorflow as tf

numClasses = 2    			# number of classes to be classified
trainSize = numClasses*4000 # number of datapoints to be trained
validSize = numClasses*1000	# number of datapoints to be validated on
testSize = numClasses*1000	# number of datapoints to be tested on
numClasses = 2
eta = 0.005 		# learning rate
sizeLayer1 = 10     # layer 1 has weight matrix of dimensions sizeLayer1 X sizeLayer1
sizeLayer2 = 5		# layer 2 has weight matrix of dimensions sizeLayer2 X sizeLayer2
sizeLayer3 = 10		# layer 3 is fully connected layer with sizeLayer3 output neurons
featLayer1 = 15		# number of features at the output of layer 1
featLayer2 = 50		# number of features at the output of layer 2
trainingSeries = np.matrix(np.load("../../cifar-10-batches-py/trainingSeries2.npy").astype('float32'))/255 # file containing all the training data points 
testSeries = np.matrix(np.load("../../cifar-10-batches-py/testSeries2.npy").astype('float32'))/255 # file containing all the test data points
trainingLabels = np.load("../../cifar-10-batches-py/trainingLabels2.npy") # labels corresponding to the training data points (one hot encoded)
testLabels = np.load("../../cifar-10-batches-py/testLabels2.npy") # labels corresponding to the test data points (one hot encoded)
trainingLabels = trainingLabels[:,0:numClasses]
testLabels = testLabels[:,0:numClasses]



numPixels = 3072 # number of pixels
numDots = 32 # dimension of the image
numChannels = 3 # channels per image



inp = tf.placeholder(tf.float32,shape = [None,numPixels]) # input image (3072 X 1) to be placed in this placeholder
inpF = tf.reshape(inp,[-1,numDots,numDots,numChannels]) # reshape input image to make it 3 X 32 X 32
expected = tf.placeholder(tf.uint8,shape = [None,numClasses]) # place the one hot encoded expected class labels here
expectedNum = tf.argmax(expected,axis = 1) # expected classified class

weightsLayer1 = tf.Variable(tf.truncated_normal([sizeLayer1,sizeLayer1,3,featLayer1],stddev = 0.1)) # layer 1 filter weight matrix randomly initialized
biasesLayer1 = tf.Variable(tf.constant(0.1,shape=[featLayer1])) # layer 1 bias vector randomly initialized
layer1 = tf.nn.conv2d(input = inpF,filter = weightsLayer1,strides = [1,1,1,1],padding='SAME') # convolve input image with layer 1 filter
layer1 += biasesLayer1 # add biases to the result
layer1 = tf.nn.max_pool(value = layer1,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME') # perform max pooling
layer1 = tf.nn.relu(layer1) # taking relu over the maxpooled output to give final output

weightsLayer2 = tf.Variable(tf.truncated_normal([sizeLayer2,sizeLayer2,featLayer1,featLayer2],stddev = 0.1))# layer 2 has the same structure as layer 1 but with the corresponding dimensions adjusted
biasesLayer2 = tf.Variable(tf.constant(0.1,shape=[featLayer2]))
layer2 = tf.nn.conv2d(input = layer1,filter = weightsLayer2,strides = [1,1,1,1],padding='SAME')
layer2 += biasesLayer2
layer2 = tf.nn.max_pool(value = layer2,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
layer2 = tf.nn.relu(layer2)
lshape2 = layer2.get_shape()

layer3in = tf.reshape(layer2,[-1,lshape2[1:4].num_elements()]) # flatten the output of layer 2 to give an input to the fully connected layer
weightsLayer3 = tf.Variable(tf.truncated_normal([lshape2[1:4].num_elements(),sizeLayer3],stddev = 0.1)) # weights of the first layer of the fcnn
biasesLayer3 = tf.Variable(tf.constant(0.1,shape=[sizeLayer3])) # biases of the first layer of the fcnn
layer3out=tf.matmul(layer3in,weightsLayer3) + biasesLayer3 # Weights . Inputs + biases
layer3out = tf.nn.relu(layer3out) # output of layer after relu

weightsLayer4 = tf.Variable(tf.truncated_normal([sizeLayer3,numClasses],stddev=0.1)) # second fully connected layer same as previous
biasesLayer4 = tf.Variable(tf.constant(0.6,shape=[numClasses]))
layer4out = tf.matmul(layer3out,weightsLayer4) + biasesLayer4
fin = tf.nn.softmax(layer4out) # softmax applied instead of relu to give final output 
finCl = tf.argmax(fin,axis = 1) # output class computed

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer4out,labels = expected) # loss calculated by comparing softmax output with one hot encoded expected output
avgLoss = tf.reduce_mean(loss) # loss averaged over all datapoints

opt = tf.train.AdamOptimizer(learning_rate=eta).minimize(avgLoss) # trained using adam optimizer to minimize average loss

correct = tf.equal(finCl,expectedNum) # compare expected labels with test outputs
acc = tf.reduce_mean(tf.cast(correct,tf.float32)) # calculate accuracy from the above

numWeights = (sizeLayer1*sizeLayer1)*numChannels*featLayer1 + (sizeLayer2*sizeLayer2)*featLayer1*featLayer2 + lshape2[1:4].num_elements()*sizeLayer3 + sizeLayer3*numClasses # number of weights calcuated
print("number of weights : " + str(numWeights)) #print the number of weights
sess = tf.Session() # declare tensorflow session
sess.run(tf.global_variables_initializer()) # initialize all variables

batchSize = 20  # batch size while training
numItr = int(trainSize/batchSize) # number of batches per epoch
valCount = 0 # keeps track of the number of times consecutive validation accuracy differed by not more than 1%
prevacc = 0 # store validation accuracy of previous epoch
while valCount <= 3:
	for i in range(numItr-1):
		xB = trainingSeries[i*batchSize:(i+1)*batchSize,:] # training data points of current batch
		yB = trainingLabels[i*batchSize:(i+1)*batchSize,:] # training data labels of current batch
		feedDict1 = {inp:xB,expected:yB} # make dictionary out of the batch by placing the data points in inp and output labels in expected
		sess.run(opt,feed_dict = feedDict1) # run optimizer over the batch to train the cnn
		if i%100==0: # after every 100 batches do validation
			xT = trainingSeries[trainSize:trainSize+1000,:] # validation data points
			yT = trainingLabels[trainSize:trainSize+1000,:]	# validation data labels
			feedDict2 = {inp:xT,expected:yT} 				# subsequent steps same as training
			accu = 100*sess.run(acc,feed_dict=feedDict2)
			print("batch "+str(i)+" - "+str(accu))
			if abs(accu-prevacc) < 1:
				valCount+=1
			prevacc = accu
			if valCount>=3: # break out of training loop if 3 consecutive rounds with same validation accuracy
				break

xTest = testSeries # Test the cnn on the testing data
yTest = testLabels
feedDict3 = {inp:xTest,expected:yTest}
accuF = sess.run(acc,feed_dict=feedDict3)
print("Final accuracy - "+str(accuF)) # print accuracy on test data
	




