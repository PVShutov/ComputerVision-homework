import numpy as np
import json
import pickle


def save(f, data):
	with open(f, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(f):
	with open(f, 'rb') as handle:
		return pickle.load(handle)


main_dir = 'E:/Data_GraduationWork/ObjectLocalization/100k'

with open(main_dir + '/data/images/images.json') as file:
	json_images = json.load(file)

for i in range(100):
	save_file_name = main_dir + '/data/images/images{0}.pickle'.format(i)
	save(save_file_name, np.array(json_images[i * 1000:(i + 1) * 1000], dtype=np.uint8))
