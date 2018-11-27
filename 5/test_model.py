import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches


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
	ClassificationTruePositives = 0
	IoUs = []

	for i in range(len(labels)):
		temp_IoU = -1
		if labels[i] == gt_labels[i]:
			temp_IoU = IoU(coords[i], gt_coords[i])
			if temp_IoU >= 0.5:
				TruePositives += 1

			ClassificationTruePositives += 1
		IoUs.append(temp_IoU)

	return TruePositives, ClassificationTruePositives, IoUs


def main():

	dataset_path = 'E:/Data_GraduationWork/ObjectLocalization/100k'
	test_dataset = mnist_localization_dataset(dataset_path, 95000, 100000)
	test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=0)


	model = torch.load("./model")
	model.eval()

	test_acc = 0.0
	classification_acc = 0.0

	min_IoU, max_IoU = 1, 0
	min_IoU_id, max_IoU_id = 0, 0
	min_IoU_coord, max_IoU_coord = None, None


	for i, batch in enumerate(test_dataloader):
		images = batch[0].cuda().view(-1, 1, 64, 64)
		labels = batch[1].cuda().long()
		coords = batch[2].numpy()
		output_classification, output_localization = model(images)
		_, output_classification = torch.max(output_classification.data, 1)
		output_classification, output_localization = output_classification.cpu().detach().numpy(), output_localization.cpu().detach().numpy()
		TruePositives, ClassificationTruePositives, IoUs = CalcTruePositives(output_classification, output_localization, labels, coords)
		test_acc += TruePositives
		classification_acc += ClassificationTruePositives

		for j, image_IoU in enumerate(IoUs):
			if image_IoU > 0 and image_IoU < min_IoU:
				min_IoU = image_IoU
				min_IoU_id = i*128 + j
				min_IoU_coord = output_localization[j]
			if image_IoU > 0 and image_IoU > max_IoU:
				max_IoU = image_IoU
				max_IoU_id = i*128 + j
				max_IoU_coord = output_localization[j]


	min_IoU_image, min_IoU_gt_coord = test_dataset[min_IoU_id][0], test_dataset[min_IoU_id][2]
	max_IoU_image, max_IoU_gt_coord = test_dataset[max_IoU_id][0], test_dataset[max_IoU_id][2]


	fig, ax = plt.subplots(1)
	x, y, w, h = 32 * (min_IoU_gt_coord[0] + 1.0), 32 * (min_IoU_gt_coord[1] + 1.0), \
				32 * (min_IoU_gt_coord[2] - min_IoU_gt_coord[0]), 32 * (min_IoU_gt_coord[3] - min_IoU_gt_coord[1]),
	rect0 = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')

	x, y, w, h = 32 * (min_IoU_coord[0] + 1.0), 32 * (min_IoU_coord[1] + 1.0), \
				32 * (min_IoU_coord[2] - min_IoU_coord[0]), 32 * (min_IoU_coord[3] - min_IoU_coord[1]),
	rect1 = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#ff66b3', facecolor='none', linestyle=':')
	ax.imshow(1.0 - (min_IoU_image + 1.0) / 2.0, cmap='Greys')
	ax.add_patch(rect0)
	ax.add_patch(rect1)
	print(min_IoU)
	plt.show()



	fig, ax = plt.subplots(1)
	x, y, w, h = 32 * (max_IoU_gt_coord[0] + 1.0), 32 * (max_IoU_gt_coord[1] + 1.0), \
				32 * (max_IoU_gt_coord[2] - max_IoU_gt_coord[0]), 32 * (max_IoU_gt_coord[3] - max_IoU_gt_coord[1]),
	rect0 = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')

	x, y, w, h = 32 * (max_IoU_coord[0] + 1.0), 32 * (max_IoU_coord[1] + 1.0), \
				32 * (max_IoU_coord[2] - max_IoU_coord[0]), 32 * (max_IoU_coord[3] - max_IoU_coord[1]),
	rect1 = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#ff66b3', facecolor='none', linestyle=':')
	ax.imshow(1.0 - (max_IoU_image + 1.0) / 2.0, cmap='Greys')
	ax.add_patch(rect0)
	ax.add_patch(rect1)
	print(max_IoU)
	plt.show()



	test_acc /= len(test_dataset)
	classification_acc /= len(test_dataset)
	print(test_acc, classification_acc)

if __name__ == '__main__':
	main()