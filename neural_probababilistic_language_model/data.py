import numpy as np


def make_batch(sentences,word_dict,number_dict,n_class):
	input_batch = []
	target_batch =[]

	for sen in sentences:
		word =sen.split()
		input =[word_dict[n] for n in word[:-1]]
		target = word_dict[word[-1]]
		input_batch.append(np.eye(n_class)[input])
		target_batch.append(target)

	return input_batch,target_batch
