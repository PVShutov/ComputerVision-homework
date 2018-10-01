import os
from urllib.request import urlretrieve
from urllib.parse import urljoin



def download_file(datasets_url, fname, target_dir):
    target_fname = os.path.join(target_dir, fname)
    if not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        urlretrieve(url, target_fname)
    return target_fname


