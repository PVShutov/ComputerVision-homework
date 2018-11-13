import torch, torchvision


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
	elif classname.find('Linear') != -1:
		torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.main = torch.nn.Sequential(
			torch.nn.Conv2d(3, 16, 3, 1, 1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(16, 16, 3, 1, 1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 16x16

			torch.nn.Conv2d(16, 32, 3, 1, 1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(32, 32, 3, 1, 1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 8x8

			torch.nn.Conv2d(32, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(64, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 4x4
		)  # 4x4s64 = 1024
		self.output = torch.nn.Sequential(
			torch.nn.Linear(1024, 256),
			torch.nn.BatchNorm1d(256),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(256, 256),
			torch.nn.BatchNorm1d(256),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(256, 10)
		)

	def forward(self, input):
		input = self.main(input)
		input = input.reshape(-1, 1024)
		return self.output(input)


def exp_lr_scheduler(optimizer, iter, lr=0.001, lr_decay_steps=500, factor=0.95):
	lr = lr * (factor ** (iter // lr_decay_steps))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer


import visdom

def main():
	transform_train = torchvision.transforms.Compose([
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
	])
	transform_test = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
	])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	vis = visdom.Visdom()

	M = Model().cuda()
	M.apply(weights_init)

	loss = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(M.parameters(), lr=0.001)

	iter_num = 1
	test_accs = [0]
	while True:
		batch = next(iter(trainloader))
		images = batch[0].cuda()
		labels = batch[1].cuda()

		exp_lr_scheduler(optimizer, iter=iter_num, lr=0.001, lr_decay_steps=100)
		optimizer.zero_grad()
		m_loss = loss(M(images), labels)
		m_loss.backward()
		optimizer.step()

		vis.line([m_loss.item()], [iter_num], 'loss', name='loss', update='append', opts=dict(title='Loss'))
		if iter_num % 10 == 0:
			M.eval()
			test_acc = 0.0
			for batch in testloader:
				images = batch[0].cuda()
				labels = batch[1].cuda()
				outputs = M(images)
				_, prediction = torch.max(outputs.data, 1)
				test_acc += torch.sum(prediction == labels.data).cpu().numpy()
			test_acc /= len(testset)


			if test_acc > min(test_accs):
				torch.save(M, './model')
			test_accs.append(test_acc)

			vis.line([test_acc], [iter_num], 'test accuracy', name='accuracy', update='append',
			         opts=dict(title='Accuracy'))
			M.train()

		iter_num += 1


if __name__ == '__main__':
	main()
