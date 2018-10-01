import numpy as np



def softmax(input):
	sum_e = np.sum(np.exp(input))
	return np.exp(input) / sum_e

def relu(input):
	return np.maximum(np.zeros_like(input), input)