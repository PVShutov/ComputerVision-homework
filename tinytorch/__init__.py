import numpy as np
from tinytorch import init, math

class Tensor():
	data=None
	grad=None
	function=None
	trainable=False
	def __init__(self, shape=None, grad_fn=None, trainable=False):
		if shape is not None:
			self.data = np.zeros(shape, dtype=float)
			self.grad = np.zeros(shape, dtype=float)
		self.function=grad_fn
		self.trainable=trainable
	def __str__(self):
		return 'Tensor( data={0}, grad={1}, grad_fn={2} )'.format(self.data, self.grad, self.function)

	def from_numpy(np_array : np.ndarray, function=None):
		A = Tensor()
		A.data = np.copy(np_array)
		A.grad = np.zeros_like(A.data)
		A.function = function
		return A

	def backward(self, outputs=None):
		if outputs is None:
			outputs = np.ones_like(self.grad)
		if self.trainable:
			self.grad += outputs
		if self.function:
			grad = self.function.backward(outputs, *self.function.args)
			self.function.inputs.backward(grad)

	def zero_grad(self):
		self.grad.fill(0.0)


class Function():
	parameters = {}
	def __call__(self, inputs, *args):
		return self.call_default(inputs, *args)
	def call_default(self, inputs, *args):
		self.inputs = inputs
		self.args = args
		return Tensor.from_numpy(self.forward(inputs, *args), self)
	def forward(self, inputs, *args):
		return inputs
	def backward(self, outputs, *args):
		return outputs

	def zero_grad(self):
		pass