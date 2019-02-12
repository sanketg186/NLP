import torch
import torch.nn as nn
import numpy as np 
from data import make_batch
from model import NNLM
import torch.optim as optim
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor


sentences = [ "i like dog", "i love coffee", "i hate milk","we play football","we watch cricket","They drink coffee","We drink water","You eat banana"]
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i,w in enumerate(word_list)}
number_dict = {i: w for i,w in enumerate(word_list)}

n_class=len(word_dict)
n_step=2 #n-1
n_hidden=2
num_epochs=40000
input_batch,target_batch = make_batch(sentences,word_dict,number_dict,n_class)

# input_batch = Variable(torch.Tensor(input_batch))
# target_batch = Variable(torch.LongTensor(target_batch))
input_batch=torch.Tensor(input_batch)
target_batch=torch.Tensor(target_batch)
input_batch = Variable(input_batch.type(dtype))
target_batch = Variable(target_batch.type(torch.cuda.LongTensor))

model = NNLM(n_step,n_class,n_hidden)
model = model.type(dtype)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#train
for epoch in range(num_epochs):
	optimizer.zero_grad()
	output = model(input_batch)
	loss_val = loss(output,target_batch)
	if epoch%1000==0:
		print(loss_val.item())
	loss_val.backward()
	optimizer.step()



predict = model(input_batch).data.max(1,keepdim=True)[1]
# print(predict)
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])