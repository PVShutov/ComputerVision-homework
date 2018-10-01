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
	tinytorch.nn.Linear(784, 800, init.Initializer(init.he_normal)),
	tinytorch.nn.ReLU(),
	tinytorch.nn.Linear(800, 10, init.Initializer(init.xavier_normal))
])
loss = tinytorch.nn.SoftmaxCrossEntropyWithLogits()


import visdom
vis = visdom.Visdom() #for loss visualization
avg_loss = 0






def get_accuracy():
	count = len(test_lables)
	right_count = 0
	for i in range(len(test_lables)):
		input.data = test_images[i].flatten()
		label = np.argmax(math.softmax(seq(input).data))
		if label == test_lables[i]:
			right_count += 1
	return(right_count/count)


iter_count = 60000
learning_rate = 0.01
for i in range(iter_count):

	input.data = train_images[i].flatten()

	label.data.fill(0.0)
	label.data[train_lables[i]] = 1.0


	seq.zero_grad()
	out = loss(seq(input), label)
	out.backward()

	for key, value in seq.parameters.items():
		value.data -= learning_rate*value.grad
	learning_rate = 0.01*(0.07**(i/iter_count))


	avg_loss += out.data
	if i % 100 == 0 and i > 0:
		avg_loss /= 100.0
		vis.line([avg_loss], [i], 'avgloss', name='AvgLoss', update='append', opts=dict(title='AvgLoss'))
		avg_loss = 0

	if i % 1000 == 0:
		vis.line([get_accuracy()], [i], 'accuracy', name='Accuracy', update='append', opts=dict(title='Accuracy'))







