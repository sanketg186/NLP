import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from model import TextCNN
import numpy as np
dtype = torch.cuda.FloatTensor
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels=[1,1,1,0,0,0]
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w:i for i,w in enumerate(word_list)}
vocab_size = len(word_dict)
inputs = []

for sen in sentences:
	# inputs.append(np.asarray([word_dict[n] for n in sen.split()]))
	inputs.append(np.eye(vocab_size)[[word_dict[n] for n in sen.split()]])

targets = []

for out in labels:
	targets.append(out)

num_classes = 2
embedding_size = 2
sequence_length = 3
filter_sizes = [2,2,2]
num_filters = 3

input_batch = Variable(torch.Tensor(inputs).type(dtype))
target_batch = Variable(torch.Tensor(targets).type(torch.cuda.LongTensor))

model = TextCNN(num_filters=num_filters,filter_sizes=filter_sizes,vocab_size=vocab_size,embedding_size=embedding_size,num_classes=num_classes,sequence_length=sequence_length)
model = model.type(dtype)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
print(input_batch.size())
num_epochs=5000
for epoch in range(num_epochs):
	optimizer.zero_grad()
	output=model(input_batch)
	loss = criterion(output,target_batch)

	if (epoch+1)%1000==0:
		print("Epoch :",loss.item())

	loss.backward()
	optimizer.step()

test_text = 'sorry love you'
# tests = [np.asarray([word_dict[n] for n in test_text.split()])]
tests = np.eye(vocab_size)[[word_dict[n] for n in test_text.split()]]
test_batch = Variable(torch.Tensor(tests).type(dtype))
print(test_batch.size())
test_batch = test_batch.unsqueeze(0)
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")