import torch.nn as nn
import numpy as np
import torch
dtype = torch.cuda.FloatTensor
class TextCNN(nn.Module):
	def __init__(self,num_filters,filter_sizes,vocab_size,embedding_size,num_classes,sequence_length):
		super(TextCNN,self).__init__()
		self.num_filters = num_filters
		self.filter_sizes = filter_sizes
		self.vocab_size = vocab_size
		self.num_filters_total = num_filters*len(filter_sizes)
		self.hid_layer = nn.Linear(vocab_size,embedding_size)
		self.conv=[None]*num_filters
		self.max_pool=[None]*num_filters
		self.hidden_layer = nn.Linear(vocab_size,embedding_size)
		for i in range(num_filters):
			self.conv[i] = nn.Conv2d(1,num_filters,kernel_size = (filter_sizes[i],embedding_size))
			self.conv[i] = self.conv[i].type(dtype)
			self.max_pool[i] = nn.MaxPool2d((sequence_length-filter_sizes[i]+1,1))
		self.output_layer = nn.Linear(self.num_filters_total,num_classes)

		self.relu = nn.ReLU()


	def forward(self,x):
		pooled_outputs = []
		# print(self.vocab_size)
		# print(x.size())
		h = self.hidden_layer(x)
		h = h.unsqueeze(1)
		# print(h.size())
		
		for i in range(self.num_filters):
			# print(i)
			c = self.conv[i](h)
			z = self.relu(c)
			# print(z.size())
			z = self.max_pool[i](z)
			# print(z.size())
			z=z.permute(0,3,2,1)
			pooled_outputs.append(z)

		h_pool = torch.cat(pooled_outputs,len(self.filter_sizes))
		h_pool_flat = torch.reshape(h_pool,[-1,self.num_filters_total])
		output = self.output_layer(h_pool_flat)

		return output





