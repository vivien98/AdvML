import numpy as np
import math
import matplotlib.pyplot as pl
import scipy as sp
import matplotlib as mpl
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(torch.nn.Module) :

	def __init__(self):
		self.conv1= nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
