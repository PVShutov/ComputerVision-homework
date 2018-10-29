import torch, torchvision, visdom
BATCH_SIZE = 16

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(torch.nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.filters = 32
		self.start_image = torch.nn.ConvTranspose2d(100, self.filters//2, 4, bias=True) #4x4
		self.start_label = torch.nn.ConvTranspose2d(10, self.filters//2, 4, bias=True)
		self.main = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(self.filters),
		                                torch.nn.ConvTranspose2d(self.filters, self.filters*2, 4, 2, 1, bias=True),  #8x8
		                                torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(self.filters*2),
		                                torch.nn.ConvTranspose2d(self.filters*2, self.filters*4, 4, 2, 1, bias=True),  #16x16
		                                torch.nn.ReLU(inplace=True), torch.nn.BatchNorm2d(self.filters*4),
										torch.nn.ConvTranspose2d(self.filters*4, 1, 4, 2, 1, bias=True))   #32x32
		self.one_hot_tensor = torch.eye(10).cuda()
	def forward(self, input, label):
		input = torch.cat([self.start_image(input.view(-1, 100, 1, 1)), self.start_label(self.one_hot_tensor[label].view(-1, 10, 1, 1))], 1)
		return torch.tanh(self.main(input))

class Discriminator(torch.nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.filters = 32
		self.start_image = torch.nn.Conv2d(1, self.filters//2, 4, 2, 1, bias=True) #16x16
		self.start_label = torch.nn.Conv2d(10, self.filters//2, 4, 2, 1, bias=True)
		self.main = torch.nn.Sequential(torch.nn.LeakyReLU(0.2, inplace=True), torch.nn.BatchNorm2d(self.filters),
		                                torch.nn.Conv2d(self.filters, self.filters*2, 4, 2, 1, bias=True),  #8x8
		                                torch.nn.LeakyReLU(0.2, inplace=True), torch.nn.BatchNorm2d(self.filters*2),
		                                torch.nn.Conv2d(self.filters*2, self.filters*4, 4, 2, 1, bias=True),  #4x4
		                                torch.nn.LeakyReLU(0.2, inplace=True), torch.nn.BatchNorm2d(self.filters*4),
										torch.nn.Conv2d(self.filters*4, 1, 4, 1, 0, bias=True))   #1x1
		self.fill_tensor = torch.zeros([10, 10, 32, 32])
		for i in range(10):
			self.fill_tensor[i, i, :, :] = 1
		self.fill_tensor = self.fill_tensor.cuda()

	def forward(self, input, label):
		input = torch.cat([self.start_image(input), self.start_label(self.fill_tensor[label])], 1)
		return torch.sigmoid(self.main(input).squeeze())

def main():
	vis = visdom.Visdom()
	dataloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST("../dataset/mnist", train=True, download=True,
		                      transform=torchvision.transforms.Compose([
			                      torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor(),
			                      torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
		                      ])), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

	G, D = Generator().cuda(), Discriminator().cuda()
	G.apply(weights_init)
	D.apply(weights_init)
	G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.99))
	D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.99))
	G_scheduler = torch.optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda=[lambda iter: 0.95 ** (iter/500.0)])
	D_scheduler = torch.optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda=[lambda iter: 0.95 ** (iter/500.0)])


	z_gen = lambda : (torch.randn((BATCH_SIZE, 100)).cuda(), torch.randint(0, 10, (BATCH_SIZE,), dtype=torch.long).cuda())

	z_fixed = torch.randn((100, 100)).cuda()
	labels_fixed = torch.LongTensor([i//10 for i in range(100)]).cuda()

	y_ones, y_zeros = torch.ones((BATCH_SIZE), device='cuda'), torch.zeros((BATCH_SIZE), device='cuda')
	BCE = torch.nn.BCELoss()
	iter = 0
	while(True):
		for batch in dataloader:

			gen = z_gen()
			D.zero_grad()
			D_loss = BCE(D(batch[0].cuda(), batch[1].long().cuda()), y_ones) + BCE(D(G(*gen), gen[1]), y_zeros)
			D_loss.backward()
			D_optimizer.step()
			D_scheduler.step()

			gen = z_gen()
			G.zero_grad()
			G_loss = BCE(D(G(*gen), gen[1]), y_ones)
			G_loss.backward()
			G_optimizer.step()
			G_scheduler.step()

			if iter % 10 == 0:
				vis.line([D_loss.item()], [iter], 'gan_loss', name='D_loss', update='append', opts=dict(title='Loss'))
				vis.line([G_loss.item()], [iter], 'gan_loss', name='G_loss', update='append', opts=dict(title='Loss'))
				vis.images(G(z_fixed, labels_fixed).cpu() * 0.5 + 0.5, nrow=10, win="generator_out", opts=dict(title='Generator out'))

			if iter % 100 == 0:
				torch.save(G, './generator')
				torch.save(D, './discriminator')
			iter += 1

if __name__ == '__main__':
	main()