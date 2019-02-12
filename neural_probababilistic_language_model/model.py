#neural network language model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
dtype = torch.cuda.FloatTensor

class NNLM(nn.Module):
	def __init__(self,n_step,n_class,n_hidden):
		super(NNLM,self).__init__()
		self.n_step=n_step
		self.n_class=n_class
		self.n_hidden=n_hidden
		self.layer1 = nn.Sequential(nn.Linear(n_step*n_class,n_hidden))
		self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_class))
		self.activate = nn.Tanh()
	def forward(self,x):
		x = x.view(-1,self.n_step*self.n_class)
		h1 = self.layer1(x)
		h2 = self.layer2(h1)
		h3 = self.activate(h2)
		return h3

