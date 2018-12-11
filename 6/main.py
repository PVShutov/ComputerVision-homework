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



def main():

	dataset_path = 'E:/Data_GraduationWork/ObjectLocalization/100k'
	test_dataset = mnist_localization_dataset(dataset_path, 95000, 100000)
	test_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=0)


	model = torch.load("./model")
	model.eval()

	localized_samples = []
	samples_count = 0
	localized_samples_count = 0



	FPNs = np.zeros(10, dtype=int)
	for i, batch in enumerate(test_dataloader):
		images = batch[0].cuda().view(-1, 1, 64, 64)
		labels = batch[1].numpy()
		coords = batch[2].numpy()
		output_classification, output_localization = model(images)
		output_classification, output_localization = torch.nn.functional.softmax(output_classification).cpu().detach().numpy(), output_localization.cpu().detach().numpy()

		samples_count += len(output_classification)
		for i in range(len(output_classification)):
			if IoU(output_localization[i], coords[i]) >= 0.5:
				localized_samples.append([output_classification[i], output_localization[i], labels[i], coords[i]])
				localized_samples_count += 1
			else:
				FPNs[labels[i]] += 1

	FPNs = list(FPNs)

	MAP = 0
	for k in range(10):
		Precisions = []
		Recalls = []
		for t in np.linspace(0.0, 1.0, 10):
			TP = 0
			FP, FN = 0, 0
			for sample in localized_samples:
				pred_c_vec = sample[0]
				gt_c = sample[2]

				pred_c = np.argmax(pred_c_vec)
				if k == gt_c:
					if pred_c_vec[k] >= t and gt_c == pred_c:
						TP += 1
					if pred_c_vec[k] < t and gt_c == pred_c:
						FN += 1
					if pred_c_vec[k] >= t and gt_c != pred_c:
						FP += 1
			Precisions.append((TP / (TP + FP + FPNs[k])) if TP + FP > 0 else 1.0)
			Recalls.append(TP / (TP + FN + FPNs[k]) if TP + FN > 0 else 1.0)

		Precisions.append(0.0)
		Recalls.append(1.0)

		Precisions = np.array(Precisions)
		Recalls = np.array(Recalls)

		ind = np.argsort(-Recalls)

		AP = 0
		last_x = 0
		last_y = 0
		for i in ind:
			AP += (last_x - Recalls[i])*(0.5*last_y + 0.5*Precisions[i])
			last_x = Recalls[i]
			last_y = Precisions[i] if last_y < Precisions[i] else last_y
		MAP += AP
	MAP/=10
	print(MAP)


if __name__ == '__main__':
	main()