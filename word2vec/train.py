import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import Word2Vec
from torch.autograd import Variable
from data import random_batch
import matplotlib.pyplot as plt
dtype = torch.cuda.FloatTensor

sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
word_list =list(set(word_sequence))  #" ".join(sentences).split()
word_dict = {w:i for i,w in enumerate(word_list)}

batch_size = 20
embedding_size=2
voc_size = len(word_list)
num_epochs=20000

skip_grams = []

for i in range(1,len(word_sequence)-1):
	target = word_dict[word_sequence[i]]
	context = [word_dict[word_sequence[i-1]],word_dict[word_sequence[i+1]]]
	for w in context:
		skip_grams.append([target,w])

model = Word2Vec(voc_size,embedding_size)
model = model.type(dtype)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
	input_batch,target_batch = random_batch(voc_size,skip_grams,batch_size)
	input_batch  = Variable(torch.Tensor(input_batch))
	input_batch=input_batch.type(dtype)
	target_batch  = Variable(torch.cuda.LongTensor(target_batch))
	optimizer.zero_grad()
	output = model(input_batch)
	loss = criterion(output,target_batch)

	if (epoch+1)%1000==0:
		print("Epoch:",loss.item())

	loss.backward()
	optimizer.step()


for i,label in enumerate(word_list):
	learned_represent = model.hidden_layer.weight.data
	# print(i," : ",learned_represent)
	x,y=float(learned_represent[0][i]),float(learned_represent[1][i])
	plt.scatter(x,y)
	plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.savefig("represent.png")

