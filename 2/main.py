import numpy as np

import tinytorch
from tinytorch import init, nn, math
from tinytorch.datasets import mnist, DataLoader



data_train = mnist.train('../dataset/')
data_test = mnist.test('../dataset/')

avg_data = np.average(data_train[0])
std_data = np.std(data_train[0])

data_train[0] = (data_train[0]-avg_data)/std_data
data_test[0] = (data_test[0]-avg_data)/std_data



BATCH_SIZE = 4

train = DataLoader(data_train, BATCH_SIZE, True)
#test = DataLoader(data_test, BATCH_SIZE, True)


input = tinytorch.Tensor.from_numpy(np.empty(784))
label = tinytorch.Tensor.from_numpy(np.empty(10))
seq = tinytorch.nn.Sequential([
	tinytorch.nn.Linear(784, 800, init.Initializer(init.he_normal)),
	tinytorch.nn.ReLU(),
	tinytorch.nn.Linear(800, 10, init.Initializer(init.xavier_normal))
])
loss = tinytorch.nn.SoftmaxCrossEntropyWithLogits()
weight_decay = tinytorch.nn.L2Norm()



import visdom
vis = visdom.Visdom() #for loss visualization


def get_accuracy():
	count = len(data_test[1])
	right_count = 0
	for i in range(len(data_test[1])):
		input.data = data_test[0][i].flatten()
		label = np.argmax(math.softmax(seq(input).data))
		if label == data_test[1][i]:
			right_count += 1
	return(right_count/count)



avg_loss = 0
learning_rate = 0.01
for iter, batch in enumerate(train):
	seq.zero_grad()

	for i in range(BATCH_SIZE):
		input.data = batch[0][i].reshape(784)
		label.data.fill(0.0)
		label.data[batch[1][i]] = 1.0

		out = loss(seq(input), label) #+ 0.01*weight_decay(seq.layer_list[0].weight) + 0.01*weight_decay(seq.layer_list[2].weight)
		out.backward()
		avg_loss += out.data



	for key, value in seq.parameters.items():
		value.data -= learning_rate*value.grad

	if iter % 10 == 0 and iter > 0:
		learning_rate = pow(learning_rate, 1.0001)

	if iter % 100 == 0 and iter > 0:
		avg_loss /= BATCH_SIZE
		avg_loss /= 100.0
		vis.line([avg_loss], [iter], 'avgloss', name='AvgLoss', update='append', opts=dict(title='AvgLoss'))
		avg_loss = 0

	if iter % 1000 == 0 and iter > 0:
		vis.line([get_accuracy()], [iter], 'accuracy', name='Accuracy', update='append', opts=dict(title='Accuracy'))


