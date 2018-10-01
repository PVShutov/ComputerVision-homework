import numpy as np



class Initializer():
	def __init__(self, init_function, **function_kwargs):
		self.init_function = init_function
		self.function_kwargs = function_kwargs
	def init(self, tensor):
		self.init_function(tensor, **self.function_kwargs)


def xavier_normal(tensor):
	shape = tensor.data.shape
	in_plus_out = shape[0] + shape[1]
	tensor.data = np.random.normal(0.0, 2.0/in_plus_out, tensor.data.shape)


def he_normal(tensor):
	shape = tensor.data.shape
	tensor.data = np.random.normal(0.0, 2.0/shape[1], tensor.data.shape)

def normal(tensor, mean=0.0, std=1.0):
	tensor.data = np.random.normal(mean, std, tensor.data.shape)

def one(tensor):
	tensor.data.fill(1.0)

def zero(tensor):
	tensor.data.fill(0.0)