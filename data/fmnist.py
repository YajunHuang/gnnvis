"""
"""
import os
import struct
from pathlib import Path

import numpy as np

from .base import BaseDataset
from .utils import download, extract_archive


__all__ =['FMNISTDataset']


class FMNISTDataset(BaseDataset):
    """ FMNIST dataset object.
    """
    
    def __init__(self, data_dir, n_samples=60000, k=30, split_rates=None):
        self.name = 'fmnist'
        super().__init__(data_dir, n_samples, k, split_rates)

    def _load_data(self):
        """ Download dataset and prepocess data to features, 
        labels and adjacency matrix of knn graph.
        """
        home_url = 'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/'
        images_url = home_url + 'train-images-idx3-ubyte.gz'
        labels_url = home_url + 'train-labels-idx1-ubyte.gz'
        compress_images_file_path = os.path.join(self.download_dir, "train-images-idx3-ubyte.gz")
        compress_labels_file_path = os.path.join(self.download_dir, "train-labels-idx1-ubyte.gz")
        download(images_url, compress_images_file_path)
        download(labels_url, compress_labels_file_path)
        images_file_path = os.path.join(self.extract_dir, "train-images-idx3-ubyte")
        labels_file_path = os.path.join(self.extract_dir, "train-labels-idx1-ubyte")
        extract_archive(compress_images_file_path, images_file_path)
        extract_archive(compress_labels_file_path, labels_file_path)
        with open(images_file_path, 'rb') as fh:
            image_buf = fh.read()
        with open(labels_file_path, 'rb') as fh:
            label_buf = fh.read()
        self.features = get_image(image_buf, self.n_samples)
        self.labels = get_label(label_buf, self.n_samples)


def get_image(data_buf, num, isinverse=False, dtype=np.float32):
    """ Extract image data from binary data.

    Parameters:
    -----------
    data_buf:
        Binary data buffer
    num: int
        Number of images extracts from data
    isinverse: bool
        Whether to inverse the normalized gray values 
    dtype: numpy numeric type
        The numeric type of gray values
    
    Return:
    -------
    images: 2D np.ndarray
        the pixel grey values of images
    """
    image_index = 0
    image_index += struct.calcsize('>IIII')
    images = np.zeros((num, 784), dtype=dtype)
    for i in range(num):
        temp = struct.unpack_from('>784B', data_buf, image_index)
        if isinverse:
            images[i, :] = 1.0 - (np.array(np.reshape(temp, (784,)), dtype=dtype) / 255.0)
        else:
            images[i, :] = np.array(np.reshape(temp, (784,)), dtype=dtype) / 255.0
        image_index += struct.calcsize('>784B')
    return images


def get_label(data_buf, num, dtype=np.int32):
    """Extract label data from binary data.

    Parameters:
    -----------
    data_buf:
        Binary data buffer
    num: int
        Number of lables extracts from data

    Return:
    -------
    labels: 1D np.ndarray
        the label values
    """
    label_index = 0
    label_index += struct.calcsize('>II')
    labels = np.zeros((num, ), dtype=dtype)
    for i in range(num):
        temp = struct.unpack_from('>1B', data_buf, label_index)
        labels[i] = temp[0]
        label_index += struct.calcsize('>1B')
    return labels
