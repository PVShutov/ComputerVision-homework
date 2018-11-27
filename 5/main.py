import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

import visdom
import pickle

def load(f):
	with open(f, 'rb') as handle:
		return pickle.load(handle)
class mnist_localization_dataset(Dataset):
	def __init__(self, main_dir, start, end):
		self.main_dir = main_dir
		self.start = start
		self.end = end
		with open(main_dir + '/data/coords/coords.json') as file:
			self.coords = np.array(json.load(file))
		with open(main_dir + '/data/labels/labels.json') as file:
			self.labels = np.array(json.load(file))
		self.images_files = [main_dir + '/data/images/images{0}.pickle'.format(i) for i in range(100)]
		self.images_file = 0
		self.active_images_file = load(self.main_dir + '/data/images/images{0}.pickle'.format(self.images_file))
		self.coords = (self.coords - 32) / 32

	def __len__(self):
		return self.end - self.start
	def __getitem__(self, idx):
		idx += self.start
		key = idx//1000
		if key != self.images_file:
			self.images_file = key
			self.active_images_file = load(self.main_dir + '/data/images/images{0}.pickle'.format(self.images_file))
		image = self.active_images_file[idx - int(key) * 1000]
		return ((image - 127.5)/127.5).astype(np.float32), self.labels[idx], (self.coords[idx].flatten()).astype(np.float32)



def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
	elif classname.find('Linear') != -1:
		torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class LocalizationModel(torch.nn.Module):
	def __init__(self):
		super(LocalizationModel, self).__init__()

		self.main_part = torch.nn.Sequential(
			torch.nn.Conv2d(1, 16, 3, 1, 1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(16, 16, 3, 1, 1),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 32x32


			torch.nn.Conv2d(16, 32, 3, 1, 1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(32, 32, 3, 1, 1),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(inplace=True),
		)

		self.second_part = torch.nn.Sequential(
			torch.nn.MaxPool2d(2, 2),  # 16x16

			torch.nn.Conv2d(32, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(64, 64, 3, 1, 1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 8x8

			torch.nn.Conv2d(64, 128, 3, 1, 1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(128, 128, 3, 1, 1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(inplace=True),
			torch.nn.MaxPool2d(2, 2),  # 4x4
		)

		self.localization_part_conv = torch.nn.Conv2d(32, 1, 3, 1, 1)
		self.localization_part = torch.nn.Sequential(
			torch.nn.Linear(1024, 32),
			torch.nn.BatchNorm1d(32),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(32, 4),
			torch.nn.Tanh()
		)

		self.classification_part = torch.nn.Sequential(
			torch.nn.Linear(2048, 32),
			torch.nn.BatchNorm1d(32),
			torch.nn.ReLU(inplace=True),
			torch.nn.Linear(32, 10)
		)


	def forward(self, input):
		output = self.main_part(input)
		output_localization = self.localization_part(self.localization_part_conv(output).view(-1, 1024))
		output_2 = self.second_part(output)
		output_classification = self.classification_part(output_2.view(-1, 2048)).view(-1, 10)
		return output_classification, output_localization

def exp_lr_scheduler(optimizer, iter, lr=0.001, lr_decay_steps=500, factor=0.95):
	lr = lr * (factor ** (iter // lr_decay_steps))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return optimizer





def IoU(rect_1, rect_2):
	dx = min(rect_1[2], rect_2[2]) - max(rect_1[0], rect_2[0])
	dy = min(rect_1[3], rect_2[3]) - max(rect_1[1], rect_2[1])
	intersection_area = 0
	if (dx >= 0) and (dy >= 0):
		intersection_area = dx * dy

	area_rect_1 = (rect_1[2] - rect_1[0])*(rect_1[3] - rect_1[1])
	area_rect_2 = (rect_2[2] - rect_2[0])*(rect_2[3] - rect_2[1])

	return intersection_area/(area_rect_1 + area_rect_2 - intersection_area)


def CalcTruePositives(labels, coords, gt_labels, gt_coords):
	TruePositives = 0
	for i in range(len(labels)):
		if labels[i] == gt_labels[i]:
			if IoU(coords[i], gt_coords[i]) >= 0.5:
				TruePositives += 1
	return TruePositives




def main():
	dataset_path = 'E:/Data_GraduationWork/ObjectLocalization/100k'

	train_dataset = mnist_localization_dataset(dataset_path, 0, 90000)
	train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0)

	valid_dataset = mnist_localization_dataset(dataset_path, 90000, 95000)
	valid_dataloader = DataLoader(valid_dataset, batch_size=128, num_workers=0)




	vis = visdom.Visdom()

	LM = LocalizationModel().cuda()
	LM.apply(weights_init)

	loss_classification = torch.nn.CrossEntropyLoss()
	loss_localization = torch.nn.SmoothL1Loss()
	optimizer = torch.optim.Adam(LM.parameters(), lr=0.001)

	iter_num = 0
	valid_accs = [0]
	while True:
		for batch in train_dataloader:
			iter_num += 1

			images = batch[0].cuda().view(-1, 1, 64, 64)
			labels = batch[1].cuda().long()
			coords = batch[2].cuda()


			output_classification, output_localization = LM(images)

			exp_lr_scheduler(optimizer, iter=iter_num, lr=0.001, lr_decay_steps=50)
			optimizer.zero_grad()

			loss = loss_classification(output_classification, labels) + loss_localization(output_localization, coords)
			loss.backward()
			optimizer.step()

			vis.line([loss.item()], [iter_num], 'loss', name='loss', update='append', opts=dict(title='Loss'))



			if iter_num % 100 == 0:
				LM.eval()
				valid_acc = 0.0
				for batch in valid_dataloader:
					images = batch[0].cuda().view(-1, 1, 64, 64)
					labels = batch[1].cuda().long()
					coords = batch[2].numpy()
					output_classification, output_localization = LM(images)
					_, output_classification = torch.max(output_classification.data, 1)
					output_classification, output_localization = output_classification.cpu().detach().numpy(), output_localization.cpu().detach().numpy()
					valid_acc += CalcTruePositives(output_classification, output_localization, labels, coords)
				print(valid_acc)
				valid_acc /= len(valid_dataset)

				if valid_acc > min(valid_accs):
					torch.save(LM, './model')
				valid_accs.append(valid_acc)

				vis.line([valid_acc], [iter_num], 'validation accuracy', name='accuracy', update='append',
				         opts=dict(title='Accuracy'))
				LM.train()


if __name__ == '__main__':
	main()
