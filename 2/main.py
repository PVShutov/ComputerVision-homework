import numpy as np

import tinytorch
from tinytorch import init, nn, math
from tinytorch.datasets import mnist



train_images = mnist.train_images('../dataset/')/255.0
train_lables = mnist.train_labels('../dataset/')

test_images = mnist.test_images('../dataset/')/255.0
test_lables = mnist.test_labels('../dataset/')



input = tinytorch.Tensor.from_numpy(np.empty(784))
label = tinytorch.Tensor.from_numpy(np.empty(10))


seq = tinytorch.nn.Sequential([
	tinytorch.nn.FC_layer(784, 800, init.Initializer(init.normal, mean=0.0, std=0.002)),
	tinytorch.nn.ReLU(),
	tinytorch.nn.FC_layer(800, 10, init.Initializer(init.normal, mean=0.0, std=0.002))
])

loss = tinytorch.nn.SoftmaxCrossEntropyWithLogits()

learning_rate = 0.01
for i in range(60000):

	input.data = train_images[i].flatten()

	label.data.fill(0.0)
	label.data[train_lables[i]] = 1.0


	seq.zero_grad()
	out = loss(seq(input), label)
	out.backward()


	for key, value in seq.parameters.items():
		value.data -= learning_rate*value.grad

	learning_rate -= 0.01/60000

	#print(out.data)



count = len(test_lables)
right_count = 0
for i in range(len(test_lables)):
	input.data = test_images[i].flatten()
	label = np.argmax(math.softmax(seq(input).data))
	if label == test_lables[i]:
		right_count += 1

print('Accuracy:', right_count/count)





