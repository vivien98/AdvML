import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import optim
import mnist
import numpy as np

trainingData, trainingLabels, testData, testLabels = mnist.load()
trainingData = trainingData / 255 > 0.5
testData = testData / 255 > 0.5
trainingData = trainingData.astype(float)
testData = testData.astype(float)

trainingSize = 60000
actualTrainingSize = 50000
validSize = 10000
testSize = 10000
numEpochs = 10

class regressionClassifier(nn.Module):
	def __init__(self):
		super(regressionClassifier,self).__init__()
		self.output = nn.Linear(784,10)

	def forward(self,x):
		x = self.output(x)
		print(str(x.size()))
		#x = F.softmax(x,dim = 1 )
		return x#nn.LogSoftmax(dim1)


print(str(np.shape(testLabels)))
trainingLabelsOHE = torch.from_numpy(np.zeros((trainingSize,10)))
trainingLabels1 = torch.from_numpy(trainingLabels)
testLabelsOHE = torch.from_numpy(np.zeros((testSize,10)))
testLabels1 = torch.from_numpy(testLabels)
trainingData1 = torch.from_numpy(trainingData)
testData1 = torch.from_numpy(testData)

for i in range(trainingSize):
	trainingLabelsOHE[i,trainingLabels[i]] = 1
	if i < testSize:
		testLabelsOHE[i,testLabels[i]] = 1
	

model = regressionClassifier().float()
lastLayer = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
opt = optim.SGD(model.parameters(),lr = 0.01,weight_decay= 1e-6, momentum = 0.9, nesterov = True)
for i in range(numEpochs):
	train_loss, valid_loss = [], []
	model.train()
	for j in range(actualTrainingSize):
		opt.zero_grad()
		dataPt = trainingData1[j]
		dataLab = trainingLabels1[j]
		print(str(dataPt) + str(dataLab))
		output = lastLayer(model(dataPt.float()))
		print(str(output))
		lossPt = loss(output,dataLab)
		lossPt.backward()
		opt.step()
		train_loss.append(loss.item())
	model.eval()
	for j in range(validSize):
		dataPt = trainingData1[actualTrainingSize+j] 
		dataLab = trainingLabels1[actualTrainingSize+j]
		output = model(dataPt.float())
		lossPt = loss(output,dataLab)
		valid_loss.append(lossPt.item())
	print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Validation Loss: ", np.mean(valid_loss))

correct = 0.0
for i in range(testSize):
	dataPt = testData1[i] 
	dataLab = testLabels1[i]
	output = model(dataPt.float())
	_,prediction = torch.max(output,1)
	pred = np.squeeze(prediction.numpy())
	if pred == testLabels[i]:
		correct = correct + 1
acc = correct/testSize
print("Test Accuracy : " + str(acc))



