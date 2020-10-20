import os
import dgl
import torch
from sklearn import preprocessing
from .mnist import *
from .fmnist import *
from .prime import *


def register_data_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help=
        "The input dataset. Can be mnist"
    )


def load_knn_dataset(args, split_part=None, data_dir='./datasets/'):
    dataset = args.dataset
    n_samples = args.dsize
    k = args.k

    if dataset == 'mnist':
        mnist = MNISTDataset(data_dir, n_samples, k)
        spg = mnist.load_sparse_graph_from_npz(split_part=split_part)
        adj_matrix = spg.adj_matrix
        features = spg.attr_matrix
        if args.normalize:
            scaler = preprocessing.StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
        return features, spg.labels, adj_matrix
    elif dataset == 'fmnist':
        fmnist = FMNISTDataset(data_dir, n_samples, k)
        spg = fmnist.load_sparse_graph_from_npz(split_part=split_part)
        adj_matrix = spg.adj_matrix
        features = spg.attr_matrix
        if args.normalize:
            scaler = preprocessing.StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
        return features, spg.labels, adj_matrix
    elif dataset == 'prime':
        prime = PRIMEDataset(data_dir, n_samples, k)
        spg = prime.load_sparse_graph_from_npz(split_part=split_part)
        adj_matrix = spg.adj_matrix
        features = spg.attr_matrix
        if args.normalize:
            scaler = preprocessing.StandardScaler()
            scaler.fit(features)
            features = scaler.transform(features)
        return features, spg.labels, adj_matrix
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
