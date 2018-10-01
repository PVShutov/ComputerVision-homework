import numpy as np
import tinytorch
from tinytorch import math





class ReLU(tinytorch.Function):
	def forward(self, inputs):
		return np.maximum(np.zeros_like(inputs.data), inputs.data)
	def backward(self, outputs):
		return np.multiply(np.maximum(np.zeros_like(self.inputs.data), self.inputs.data), outputs)



class Softmax(tinytorch.Function):
	def forward(self, inputs):
		return math.softmax(inputs.data)
	def backward(self, outputs):
		y = math.softmax(self.inputs.data)
		mat = np.fill_diagonal(np.outer(-y, y), y - y*y).T
		return np.matmul(mat, outputs)


class SoftmaxCrossEntropyWithLogits(tinytorch.Function):
	def __call__(self, inputs, labels):
		return self.call_default(inputs, labels)
	def forward(self, inputs, labels):
		return -np.dot(labels.data, np.log(math.softmax(inputs.data)))
	def backward(self, outputs, labels):
		return math.softmax(self.inputs.data) - labels.data


class FC_layer(tinytorch.Function):
	def __init__(self, input_size, output_size, initializer=None):
		self.input_size = input_size
		self.output_size = output_size
		self.weight = tinytorch.Tensor((output_size, input_size), trainable=True)
		if initializer:
			initializer.init(self.weight)
		self.bias = tinytorch.Tensor(output_size, trainable=True)
		self.parameters = {'weight': self.weight, 'bias': self.bias}
	def forward(self, inputs):
		return np.dot(self.weight.data, inputs.data) + self.bias.data
	def backward(self, outputs):
		self.bias.grad = outputs
		self.weight.grad = np.outer(outputs, self.inputs.data)
		return np.dot(self.weight.data.transpose(), outputs)
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

	def forward(self, inputs):
		output = inputs
		for layer in self.layer_list:
			output = layer(output)
		return output.data

	def backward(self, outputs):
		grad = outputs
		for layer in reversed(self.layer_list):
			grad = layer.backward(grad)
		return grad

	def zero_grad(self):
		for key, value in self.parameters.items():
			value.zero_grad()