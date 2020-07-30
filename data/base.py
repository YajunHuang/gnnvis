import os
import struct
from pathlib import Path

import numpy as np

from knn import construct_sparse_knn_graph, \
                SparseGraph, save_sparse_graph_to_npz, load_sparse_graph_from_npz
from .utils import make_split_masks


__all__ =['BaseDataset']


class BaseDataset(object):
    def __init__(self, data_dir, n_samples, k, split_rates=None):
        self.data_dir = os.path.join(data_dir, self.name)
        self.n_samples = n_samples
        self.k = k
        self.masks = None
        if split_rates is not None:
            self.split_rates = split_rates
            self.masks = make_split_masks(num_samples=self.n_samples, 
                                          split_rates = self.split_rates)
        self.features = None
        self.labels = None
        self.download_dir = self.data_dir
        self.extract_dir = self.data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def make_sparse_graph_npz(self, split_part=None):
        """ Make SparseGraph instance and save to npz file for this dataset.
        """
        if self.masks is None and split_part is not None:
            raise ValueError("Split_rates has not been assigned.")
        if self.masks is not None and split_part is not None \
            and split_part >= len(self.masks):
            raise ValueError("The split part exceeds the length of split rates.")

        if self.features is None or self.labels is None:
            self._load_data()
        if self.masks is None:
            self.adj_matrix = construct_sparse_knn_graph(self.features, k=self.k)
            sparse_graph = SparseGraph(adj_matrix=self.adj_matrix, attr_matrix=self.features, labels=self.labels, metadata=self.name)
            npz_file_path = os.path.join(self.data_dir, f'{self.name}_n{self.n_samples}_k{self.k}.npz')
            save_sparse_graph_to_npz(npz_file_path, sparse_graph)
        else:
            for i, mask in enumerate(self.masks):
                if split_part is None or i == split_part:
                    features = self.features[~mask]
                    labels = self.labels[~mask]
                    adj_matrix = construct_sparse_knn_graph(features, k=self.k)
                    sparse_graph = SparseGraph(adj_matrix=adj_matrix, attr_matrix=features, labels=labels, metadata=self.name)
                    npz_file_path = os.path.join(self.data_dir, f'{self.name}_n{self.n_samples}_k{self.k}_p{i}.npz')
                    print("save to %s" %(npz_file_path))
                    save_sparse_graph_to_npz(npz_file_path, sparse_graph)

    def load_sparse_graph_from_npz(self, split_part=None):
        """ Load SparseGraph instance from npz file for efficiency.
        """
        if split_part is None:
            npz_file_path = os.path.join(self.data_dir, f'{self.name}_n{self.n_samples}_k{self.k}.npz')
        elif isinstance(split_part, int):
            npz_file_path = os.path.join(self.data_dir, f'{self.name}_n{self.n_samples}_k{self.k}_p{split_part}.npz')
        else:
            raise ValueError("unkonwn split_part type, must be None or int.")
        sparse_graph = load_sparse_graph_from_npz(npz_file_path)
        return sparse_graph

    def get_dataset(self, split_part=None):
        if self.masks is None and split_part is not None:
            raise ValueError("Split_rates has not been assigned.")
        if self.masks is not None and split_part is not None \
            and split_part >= len(self.masks):
            raise ValueError("The split part exceeds the length of split rates.")

        if self.features is None or self.labels is None:
            self._load_data()

        if self.masks is None or split_part is None:
            return self.features, self.labels
        else:
            return self.features[~mask], self.labels[~mask]
