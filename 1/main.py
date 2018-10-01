import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


def get_filter(kernel_size=3):
	kernel = np.empty((kernel_size, kernel_size))
	std2sqr = 2.0*np.sqrt(2.0)/3.0
	sum = 0
	for i, y in enumerate(np.linspace(-1.0, 1.0, kernel_size)):
		for j, x in enumerate(np.linspace(-1.0, 1.0, kernel_size)):
			kernel[i][j] = np.exp(-(x*x+y*y)/std2sqr)
			sum += kernel[i][j]
	return kernel/sum

def edge_padding(image, padding_size=2):
	padding_size = int(padding_size)
	new_image = np.empty((image.shape[0]+padding_size*2, image.shape[1]+padding_size*2))

	for i in range(padding_size):
		new_image[i, padding_size:-padding_size] = image[0]
		new_image[-padding_size + i, padding_size:-padding_size] = image[-1]
		new_image[padding_size:-padding_size, i] = image[..., 0]
		new_image[padding_size:-padding_size, -padding_size + i] = image[..., -1]
	new_image[0:padding_size, 0:padding_size] = np.mean(image[1:padding_size, 0] + image[0, 1:padding_size])/2
	new_image[0:padding_size, -padding_size:] = np.mean(image[1:padding_size, -1] + image[0, -padding_size-1:-2]) / 2
	new_image[-padding_size:, 0:padding_size] = np.mean(image[-padding_size-1:-2, 0] + image[-1, 1:padding_size]) / 2
	new_image[-padding_size:, -padding_size:] = np.mean(image[-padding_size-1:-2, -1] + image[-1, -padding_size-1:-2]) / 2
	new_image[padding_size:-padding_size, padding_size:-padding_size] = image
	return new_image


def set_filter(image, filter):
	kernel_half_size = np.ceil(filter.shape[0]/2.0)
	new_image = edge_padding(image, kernel_half_size)
	kernel_size = filter.shape[0]
	out_image = np.empty_like(image)
	for i in range(out_image.shape[0]):
		for j in range(out_image.shape[1]):
			out_image[i, j] = np.sum(np.multiply(new_image[i:i+kernel_size, j:j+kernel_size], filter))
	return out_image


def image_to_grayscale(image):
	return np.sum(image, axis=2)/3.0

def add_noise_to_image(image):
	return np.clip(image + np.random.normal(0, 1, image.shape), 0.0, 1.0)


#Загрузка картинки + перевод в [0.0, 1.0]
image = np.float32(misc.imread('../CV-2018/Materials/lena_color_512.tif'))
image = image/255.0


plt.imshow(image)
plt.title('Исходная картинка')
plt.show()


#Добавление шума к картинке
image = add_noise_to_image(image)
imgplot = plt.imshow(image, cmap="gray")
plt.title('Шумная картинка')
plt.show()


#Как работает padding:
plt.imshow(edge_padding(image_to_grayscale(image), 50), cmap="gray")
plt.title('Пример расширения картинки')
plt.show()



#Сглаживание картинки в градациях серого
filter = get_filter(9)
image = set_filter(image_to_grayscale(image), filter)

plt.imshow(image, cmap="gray")
plt.title('Сглаживание')
plt.show()


count, bins, ignored = plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
plt.title('Гистограмма цветов')
plt.show()