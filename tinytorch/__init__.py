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

	def backward(self, grad=None):
		if grad is None:
			grad = np.ones_like(self.grad)
		if self.trainable:
			self.grad += grad
		if self.function:
			self.function.base_backward(grad, *self.function.args)

	def zero_grad(self):
		self.grad.fill(0.0)



	def __add__(self, other):
		return Sum()(self, other)
	def __mul__(self, other):
		return Multiply()(self, other)
	def __truediv__(self, other):
		return Divide()(self, other)

	__radd__ = __add__
	__rmul__ = __mul__



class Function():
	parameters = {}
	def __call__(self, *args):
		return self.call_default(*args)
	def call_default(self, *args):
		self.args = args
		return Tensor.from_numpy(self.forward(*args), self)

	def forward(self, *args):
		return args
	def backward(self, grad, *args):
		return grad

	def base_backward(self, grad, *args):
		grad = self.backward(grad, *args)
		for arg in self.args:
			if type(arg) is Tensor:
				arg.backward(grad)

	def zero_grad(self):
		pass


class Sum(Function):
	def __call__(self, input1, input2):
		return self.call_default(input1, input2)
	def forward(self, input1, input2):
		return (input1.data if type(input1) is Tensor else input1) + (input2.data if type(input2) is Tensor else input2)
	def backward(self, grad,  *args):
		return grad


class Multiply(Function):
	def __call__(self, input1, input2):
		return self.call_default(input1, input2)
	def forward(self, input1, input2):
		return np.multiply(input1.data if type(input1) is Tensor else input1,
		                 input2.data if type(input2) is Tensor else input2)
	def backward(self, grad,  input1, input2):
		return grad
	def base_backward(self, grad, input1, input2):
		grad = self.backward(grad, input1, input2)
		input1_data = input1.data if type(input1) is Tensor else input1
		input2_data = input2.data if type(input2) is Tensor else input2
		if type(input1) is Tensor:
			input1.backward(np.multiply(grad, input2_data))
		if type(input2) is Tensor:
			input2.backward(np.multiply(grad, input1_data))

class Divide(Function):
	def __call__(self, input1, input2):
		return self.call_default(input1, input2)
	def forward(self, input1, input2):
		return np.divide(input1.data if type(input1) is Tensor else input1,
		                 input2.data if type(input2) is Tensor else input2)
	def backward(self, grad,  input1, input2):
		return grad
	def base_backward(self, grad, input1, input2):
		grad = self.backward(grad, input1, input2)
		input1_data = input1.data if type(input1) is Tensor else input1
		input2_data = input2.data if type(input2) is Tensor else input2
		if type(input1) is Tensor:
			input1.backward(np.divide(grad, input2_data))
		if type(input2) is Tensor:
			input2.backward(np.divide(-np.multiply(grad, input1_data), input1_data**2))