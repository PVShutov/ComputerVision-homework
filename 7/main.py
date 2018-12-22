import matplotlib.pyplot as plt, matplotlib.transforms
import numpy as np
from scipy import misc, signal



def image_to_grayscale(image):
	return np.sum(image, axis=2)/3.0


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




def border_selection(rgb_image):
	image = image_to_grayscale(rgb_image/255.0)
	filter = get_filter(5)
	image = set_filter(image, filter)

	sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
	sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])


	#image_x = signal.convolve2d(image, sobel_x, mode='same')
	image_x = set_filter(image, sobel_x)
	image_y = set_filter(image, sobel_y)

	#image_x += image_x.min()
	#image_x /= image_x.max()
	grad = np.stack((image_x, image_y), axis = 2)

	grad_mod = np.sqrt(image_x*image_x + image_y*image_y)
	grad_ang = np.arctan2(image_y, image_x)


	vectors = np.array(
		[[[1, -1], [1,0], [1,1]],
		[[0, -1], [0, 0], [0, 1]],
		[[-1, -1], [-1, 0], [-1, 1]]]
	)


	for i in range(1, image.shape[0]-1):
		for j in range(1, image.shape[1]-1):
			k_x = -1 if image_x[i, j] >= 0 else 1
			k_y = -1 if image_y[i, j] >= 0 else 1
			s_x = np.abs(image_x[i, j])
			s_y = np.abs(image_y[i, j])
			s_d = grad_mod[i, j]
			s = s_x + s_y + s_d
			test = s_x*grad_mod[i, j+k_x]+s_y*grad_mod[i+k_y, j]+s_d*grad_mod[i+k_y, j+k_x]
			if grad_mod[i, j] < test/s:
				grad_mod[i, j] = 0

	return grad_mod > 0.3





def find_lines(bin_image):

	height, width = bin_image.shape
	max_p = np.sqrt(width*width + height*height)
	acc_size = int(max_p)

	n = acc_size//20  #local-maximum-size
	n += 1-n%2
	h_n = n//2
	accumulator = np.zeros((acc_size + 2*h_n, acc_size + 2*h_n), dtype=int)
	thetas = np.linspace(-np.pi/2, np.pi, num=acc_size)
	sin_thetas, cos_thetas = np.sin(thetas), np.cos(thetas)

	#Build accumulator
	for i in range(height):
		for j in range(width):
			if bin_image[i,j]:
				p = i*sin_thetas + j*cos_thetas
				x_ind = np.argwhere(p >= 0).flatten()
				y_ind = np.round(p[x_ind]).astype(int)
				accumulator[y_ind + h_n, x_ind + h_n] += 1

	min_point = np.max(accumulator)/2
	out_lines = []


	#NMS
	for i in range(h_n, acc_size + h_n):
		for j in range(h_n, acc_size + h_n):
			temp_acc = accumulator[i - h_n:i + h_n + 1, j - h_n:j + h_n + 1]
			arg = np.argmax(temp_acc)
			if arg == h_n*n + h_n and accumulator[i, j] > min_point:
				accumulator[i, j] = 0
				out_lines.append((i - h_n, thetas[j - h_n]))

	return out_lines





image = misc.imread('../CV-2018/Materials/empire.jpg')
plt.imshow(image)
plt.show()
image_final = border_selection(np.float32(image))
plt.imshow(image_final, cmap='gray')
plt.show()


plt.imshow(image)
lines = find_lines(image_final)



height, width, chanels = image.shape
for p, theta in lines:
	x0, x1 = 0, width
	y0, y1 = p - x0*np.cos(theta), p - x1*np.cos(theta)
	y0 /= np.sin(theta)
	y1 /= np.sin(theta)

	plt.plot([x0,x1], [y0,y1], color='black', linewidth=1)


plt.xlim(0, width)
plt.ylim(height, 0)

plt.title('Image + detected lines')
plt.show()

