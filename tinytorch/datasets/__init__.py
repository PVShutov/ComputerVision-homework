import os
from urllib.request import urlretrieve
from urllib.parse import urljoin
import numpy as np


def download_file(datasets_url, fname, target_dir):
    target_fname = os.path.join(target_dir, fname)
    if not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        urlretrieve(url, target_fname)
    return target_fname



class DataLoader:

    def __init__(self, data, batch_size=1, shuffle=False):
        self.data = data
        if self.data is np.ndarray:
            self.data = [self.data]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(data[0])

    def __iter__(self):
        start = 0
        end = start + self.batch_size
        while start < self.data_len:
            out = []
            for i in range(len(self.data)):
                out.append(self.data[i][start: end if end <= self.data_len else self.data_len])
            start = end
            end += self.batch_size
            yield out

