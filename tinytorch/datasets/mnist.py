#Partly taken from https://github.com/datapythonista/mnist


import array, gzip, numpy, os, struct

datasets_url = 'http://yann.lecun.com/exdb/mnist/'

import tinytorch.datasets

def download_and_parse_mnist_file(fname, target_dir=None):
	fname = tinytorch.datasets.download_file(datasets_url, fname, target_dir=target_dir)
	fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
	with fopen(fname, 'rb') as fd:
		return parse_idx(fd)


def parse_idx(fd):
	DATA_TYPES = {0x08: 'B',  # unsigned byte
	              0x09: 'b',  # signed byte
	              0x0b: 'h',  # short (2 bytes)
	              0x0c: 'i',  # int (4 bytes)
	              0x0d: 'f',  # float (4 bytes)
	              0x0e: 'd'}  # double (8 bytes)

	header = fd.read(4)
	zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
	data_type = DATA_TYPES[data_type]
	dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
	                                fd.read(4 * num_dimensions))

	data = array.array(data_type, fd.read())
	data.byteswap()
	return numpy.array(data).reshape(dimension_sizes)


def train_images(target_dir):
	return download_and_parse_mnist_file('train-images-idx3-ubyte.gz', target_dir)


def test_images(target_dir):
	return download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', target_dir)


def train_labels(target_dir):
	return download_and_parse_mnist_file('train-labels-idx1-ubyte.gz', target_dir)


def test_labels(target_dir):
	return download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', target_dir)
