import numpy as np
import tinytorch
from tinytorch import math





class ReLU(tinytorch.Function):
	def forward(self, input):
		return math.relu(input.data)
	def backward(self, grad, input):
		return np.multiply(input.data > 0, grad)


class Softmax(tinytorch.Function):
	def forward(self, input):
		return math.softmax(input.data)
	def backward(self, grad, input):
		y = math.softmax(input.data)
		mat = np.fill_diagonal(np.outer(-y, y), y - y*y).T
		return np.matmul(mat, grad)



class L2Norm(tinytorch.Function):

	def __call__(self, input):
		return self.call_default(input)
	def forward(self, input):
		return np.sqrt(np.sum(np.square(input.data)))

	def backward(self, grad, input):
		return (input.data/(2.0*self.forward(input)))*grad


class SoftmaxCrossEntropyWithLogits(tinytorch.Function):
	def __call__(self, input, label):
		return self.call_default(input, label)
	def forward(self, input, label):
		return -np.dot(label.data, np.log(math.softmax(input.data)))
	def backward(self, grad, input, label):
		return math.softmax(input.data) - label.data




class Linear(tinytorch.Function):
	def __init__(self, input_size, output_size, initializer=None):
		self.input_size = input_size
		self.output_size = output_size
		self.weight = tinytorch.Tensor((output_size, input_size), trainable=True)
		if initializer:
			initializer.init(self.weight)
		self.bias = tinytorch.Tensor(output_size, trainable=True)
		self.parameters = {'weight': self.weight, 'bias': self.bias}
	def forward(self, input):
		return np.dot(self.weight.data, input.data) + self.bias.data
	def backward(self, grad, input):
		self.bias.grad += grad
		self.weight.grad += np.outer(grad, input.data)
		return np.dot(self.weight.data.transpose(), grad)
	def zero_grad(self):
		self.weight.zero_grad()
		self.bias.zero_grad()


class Sequential(tinytorch.Function):
	def __init__(self, layer_list):
		self.layer_list = layer_list
		self.parameters = {}
		for i, layer in enumerate(layer_list):
			if len(layer.parameters) > 0:
				for key, value in layer.parameters.items():
					self.parameters['{0}.{1}'.format(i, key)] = value
	def forward(self, input):
		output = input
		for layer in self.layer_list:
			output = layer(output)
		return output.data
	def backward(self, grad, input):
		for layer in reversed(self.layer_list):
			grad = layer.backward(grad, *layer.args)
		return grad
	def zero_grad(self):
		for key, value in self.parameters.items():
			value.zero_grad()