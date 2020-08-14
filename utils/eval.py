""" """
import os
import time

import numpy as np
import torch

from .plot import scatter
from .metrics import compute_metrics
from data import load_knn_dataset


def evaluate(model, infer_model, g, features, labels, iscuda=False, ks=None, mask=None,
             eval_metric_path=None, eval_data_path=None, eval_scatter_path=None):
    for infer_param, param in zip(infer_model.parameters(), model.parameters()):
        infer_param.data.copy_(param.data)
    infer_model.eval()
    with torch.no_grad():
        logits = infer_model(g, iscuda)
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
            features = features[mask]
        logits = logits.cpu().numpy() if iscuda else logits.numpy()
        if ks:
            E, T, L = compute_metrics(features, logits, labels, ks)
            if eval_metric_path:
                with open(eval_metric_path, 'w') as fh:
                    fh.write(','.join([str(k) for k in ks]) + '\n')
                    fh.write(','.join([str(e) for e in E]) + '\n')
                    fh.write(','.join([str(t) for t in T]) + '\n')
                    fh.write(','.join([str(l) for l in L]) + '\n')
        if eval_scatter_path:
            scatter(logits, labels, eval_scatter_path)
        if eval_data_path:
            save_result(logits, eval_data_path)


def save_result(embeddings, features=None, labels=None, filepath='./eval_result.npz'):
    data_dict = {}
    data_dict['embedding'] = embeddings
    if features is not None:
        data_dict['feature'] = features
    if labels is not None:
        data_dict['label'] = labels
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    np.savez(filepath, **data_dict)


def plot_from_file(input_filepath, image_filepath):
    with np.load(input_filepath) as loader:
        loader = dict(loader)
        embeddings = loader['embedding']
        labels = loader['label']
        scatter(embeddings, labels, image_filepath)


def eval_from_file(input_filepath):
    with np.load(input_filepath) as loader:
        loader = dict(loader)
        features = loader['feature']
        embeddings = loader['embedding']
        labels = loader['label']
        E, T, L = compute_metrics(features, logits, labels, ks)
        print(E)
        print(T)
        print(L)


if __name__ == '__main__':
    dataset = load_knn_dataset('mnist', n_samples=10000, k=20)
    with np.load(
            '/Users/yale/Projects/research/gnnvis/result/mnist/2020-02-02-201658/epoch_200/eval_data.npz') as loader:
        loader = dict(loader)
        features = loader['feature']
        embeddings = loader['embedding']
        labels = loader['label']
        E, T, L = compute_metrics(dataset.features, embeddings, labels, ks=[1, 5, 10])
        print(E)
        print(T)
        print(L)
