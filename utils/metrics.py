import numpy as np
import torch

import coranking
from coranking.metrics import trustworthiness, continuity, LCMC
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances


def compute_metrics(features, embeddings, labels, ks=[1, 5, 10, 20, 30, 40, 50]):
    N = features.shape[0]
    train_mask = np.zeros(N).astype(int)
    val_mask = np.zeros(N).astype(int)
    train_mask[:int(N * 0.7)] = 1
    val_mask[int(N * 0.7):] = 1
    knn_error = knn_prediction(embeddings, labels, val_mask, train_mask, ks=ks)
    T, L = low_dimension_evaluate(features, embeddings, ks=ks)
    return knn_error, T, L


def qmatrix(high_data, low_data):
    """Generate a co-ranking matrix from two data frames of high and low
    dimensional data.

    :param high_data: DataFrame containing the higher dimensional data.
    :param low_data: DataFrame containing the lower dimensional data.
    :returns: the co-ranking matrix of the two data sets.
    """
    n, m = high_data.shape
    high_distance = distance.squareform(distance.pdist(high_data))
    low_distance = distance.squareform(distance.pdist(low_data))

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)
    del high_distance, low_distance
    Q, _, _ = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)
    return Q[1:, 1:]


def low_dimension_evaluate(high_dim_data: np.array, low_dim_data: np.array, ks: list=[1, 5, 10, 20, 30, 40, 50, 90, 100]):
    Q = qmatrix(high_dim_data, low_dim_data)
    T = trustworthiness(Q, min_k=1, max_k=101)
    T = T[np.array(ks) - 1]
    L = LCMC(Q, min_k=1, max_k=101)
    L = L[np.array(ks) - 1]
    return T, L


def knnerr(model, features, labels, val_mask, train_mask, cuda, ks=[1, 10, 20]):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if cuda:
            train_X = logits[train_mask].cpu().numpy()
            train_labels = labels[train_mask].cpu().numpy()
            test_X = logits[val_mask].cpu().numpy()
            test_labels = labels[val_mask].cpu().numpy()
        else:
            train_X = logits[train_mask].numpy()
            train_labels = labels[train_mask].numpy()
            test_X = logits[val_mask].numpy()
            test_labels = labels[val_mask].numpy()

        sum_train = np.sum(train_X ** 2, 1)
        sum_test = np.sum(test_X ** 2, 1)
        D2 = sum_train + (-2 * np.dot(train_X, test_X.T) + sum_test).T
        err = np.zeros(len(ks))
        n = test_X.shape[0]

        for i in range(n):
            tmp_D = np.argsort(D2[i, :])
            for j in range(len(ks)):
                tmp = tmp_D[:ks[j]]
                tmp_l = train_labels[tmp]
                tu = sorted([(np.sum(tmp_l == i), i) for i in set(tmp_l.flat)])
                tmplabels = tu[-1][1]
                if test_labels[i] != tmplabels:
                    err[j] += 1

        for j in range(len(ks)):
            err[j] = float(err[j]) / float(n)
        return err


# def knn_prediction(embeddings, labels, val_mask, train_mask, ks=[1, 10, 20]):
#     train_X = embeddings[train_mask]
#     train_labels = labels[train_mask]
#     test_X = embeddings[val_mask]
#     test_labels = labels[val_mask]

#     sum_train = np.sum(train_X ** 2, 1)
#     sum_test = np.sum(test_X ** 2, 1)
#     D2 = sum_train + (-2 * np.dot(train_X, test_X.T) + sum_test).T
#     err = np.zeros(len(ks))
#     n = test_X.shape[0]

#     for i in range(n):
#         tmp_D = np.argsort(D2[i, :])
#         for j in range(len(ks)):
#             tmp = tmp_D[:ks[j]]
#             tmp_l = train_labels[tmp]
#             tu = sorted([(np.sum(tmp_l == i), i) for i in set(tmp_l.flat)])
#             tmplabels = tu[-1][1]
#             if test_labels[i] != tmplabels:
#                 err[j] += 1

#     for j in range(len(ks)):
#         err[j] = float(err[j]) / float(n)
#     return err

def knn_prediction(logits, labels, val_mask, train_mask, ks=[1, 10, 20]):

    train_X = logits[train_mask]
    test_X = logits[val_mask]
    train_labels = labels[train_mask]
    test_labels = labels[val_mask]

    D2 = pairwise_distances(X=train_X, Y=test_X, metric='euclidean', n_jobs=-1) ** 2

    n = test_X.shape[0]
    err = np.zeros(len(ks))

    for i in range(n):
        tmp_D = np.argsort(D2[i, :])
        for j in range(len(ks)):
            tmp = tmp_D[:ks[j]]
            tmp_l = train_labels[tmp]
            tu = sorted([(np.sum(tmp_l == i), i) for i in set(tmp_l.flatten())])
            tmplabels = tu[-1][1]
            if test_labels[i] != tmplabels:
                err[j] += 1

    for j in range(len(ks)):
        err[j] = float(err[j]) / float(n)
    # print("\t".join(["{:.4f}".format(1 - erri) for erri in err]))
    return err