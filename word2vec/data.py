import numpy as np
def random_batch(voc_size,data,size):
	random_inputs = []
	random_labels = []
	random_index = np.random.choice(range(len(data)),size,replace=False)

	for i in random_index:
		random_inputs.append(np.eye(voc_size)[data[i][0]])
		random_labels.append(data[i][1])

	return random_inputs,random_labels


