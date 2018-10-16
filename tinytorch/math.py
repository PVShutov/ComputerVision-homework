import numpy as np



def softmax(input):
	e = np.exp(input)
	return e / e.sum()

def relu(input):
	return input * (input > 0) #np.maximum(input, 0)
