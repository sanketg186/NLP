import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class Word2Vec(nn.Module):
	def __init__(self,input_size,embedding_size):
		super(Word2Vec,self).__init__()
		self.hidden_layer = nn.Linear(input_size,embedding_size)
		self.output_layer =  nn.Linear(embedding_size,input_size)
		# self.activate = nn.Softmax()

	def forward(self,x):
		hidden = self.hidden_layer(x)
		output = self.output_layer(hidden)
		return output
