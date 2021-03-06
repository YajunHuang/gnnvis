import time
import numpy as np
import torch

import coranking
from coranking.metrics import trustworthiness, continuity, LCMC
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import multiprocessing as mp


def compute_metrics(features, embeddings, labels, ks=[1, 5, 10, 20, 30, 40, 50]):
    N = features.shape[0]
    train_mask = np.zeros(N).astype(bool)
    val_mask = np.zeros(N).astype(bool)
    train_mask[:int(N * 0.7)] = True
    val_mask[int(N * 0.7):] = True
    # knn_error = knn_prediction(embeddings, labels, val_mask, train_mask, ks=ks)
    knn_error = knn_prediction_fast(embeddings, labels, val_mask, train_mask, Ks=ks, block=-5)
    if N <= 10000:
        T, L = low_dimension_evaluate(features, embeddings, ks=ks)
    elif N <= 15000:
        T, L = low_dimension_evaluate_large(features, embeddings, ks=ks, block=10000)
    else:
        T, L = np.zeros(len(ks)), np.zeros(len(ks))
    return knn_error, T, L

def compute_test_metrics(train_features, train_embeddings, train_labels, test_features, test_embeddings, test_labels, ks=[1, 5, 10, 20, 30, 40, 50]):
    TN = train_features.shape[0]
    N = test_features.shape[0]
    train_mask = np.zeros(TN + N).astype(bool)
    val_mask = np.zeros(TN + N).astype(bool)
    train_mask[:TN] = True
    val_mask[TN:] = True
    embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
    labels = np.concatenate((train_labels, test_labels))
    knn_error = knn_prediction(embeddings, labels, val_mask, train_mask, ks=ks)
    if N <= 20000:
        T, L = low_dimension_evaluate(test_features, test_embeddings, ks=ks)
    else:
        T, L = low_dimension_evaluate_large(test_features, test_embeddings, ks=ks, block=10000)
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

def low_dimension_evaluate_large(high_dim_data: np.array, low_dim_data: np.array, ks: list=[1, 5, 10, 20, 30, 40, 50, 90, 100], block:int=10, n_jobs:int=-1):
    i = 0
    T = trustworthiness_large(high_dim_data, low_dim_data, ks, block=block, n_jobs=n_jobs)
    L = LCMC_large(high_dim_data, low_dim_data, ks, block=block, n_jobs=n_jobs)
    # print("\t".join(["{:.4f}".format(Ti) for Ti in T]))
    # print("\t".join(["{:.4f}".format(Li) for Li in L]))
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

    D2 = pairwise_distances(X=test_X, Y=train_X, metric='euclidean', n_jobs=-1) ** 2

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


def knn_prediction_fast(logits, labels, val_mask, train_mask, Ks: list = [1, 10, 20], block: int = 10000, n_jobs: int = -1):
    train_X = logits[train_mask]
    test_X = logits[val_mask]
    train_labels = labels[train_mask]
    test_labels = labels[val_mask]

    if n_jobs == -1:
        n_jobs = int(mp.cpu_count() * 0.9)
    if block < 0:
        block = n_jobs * 100 * (-block)

    n = test_X.shape[0]
    err = np.zeros(len(Ks))
    n_blocks = n // block + 1
    # print(n)
    for b in range(n_blocks):
        print("Run in block [{}/{}]".format(b, n_blocks), end="")
        start_time = time.time()
        low_X = int(b * n / n_blocks)
        high_X = int((b + 1) * n / n_blocks)
        if high_X > n:
            high_X = n
        # print("low_X: {}, high_X: {}".format(low_X, high_X))
        num_samples = test_X[low_X: high_X, :].shape[0]

        D2 = pairwise_distances(X=test_X[low_X: high_X, :], Y=train_X, metric='euclidean', n_jobs=n_jobs) ** 2

        test_labels_block = test_labels[low_X: high_X]
        def worker(id, return_dict):
            low_pos = int(id * num_samples / n_jobs)
            high_pos = int((id + 1) * num_samples / n_jobs)
            if high_pos > num_samples:
                high_pos = num_samples
            # print("low_pos: {}, high_pos: {}".format(low_pos, high_pos))
            inner_err = np.zeros(len(Ks))
            for ii in range(low_pos, high_pos):
                tmp_D = np.argsort(D2[ii, :])
                for j in range(len(Ks)):
                    tmp = tmp_D[:Ks[j]]
                    tmp_l = train_labels[tmp]
                    tu = sorted([(np.sum(tmp_l == i), i) for i in set(tmp_l.flatten())])
                    tmplabels = tu[-1][1]
                    if test_labels_block[ii] != tmplabels:
                        inner_err[j] += 1
            return_dict[id] = inner_err

        record = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for i in range(n_jobs):
            process = mp.Process(target=worker, args=(i, return_dict))
            process.start()
            record.append(process)

        for process in record:
            process.join()


        for PERROR in return_dict.values():
            err += PERROR

        # print([[p, p.exitcode, p.pid] for p in record], end="")

        del D2
        del test_labels_block
        del return_dict
        del manager
        print(" | Use time: {:.2f}s".format(time.time()-start_time))

    for j in range(len(Ks)):
        err[j] = float(err[j]) / float(n)
    # print("\t".join(["{:.4f}".format(1 - erri) for erri in err]))
    return err


def LCMC_large(X:np.array, mappedX:np.array, Ks:list=[1], block:int=10, n_jobs:int=-1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n, d = X.shape
    i_s = [block] * int(n / block) if n % block == 0 else [block] * int(n / block) + [n % block]
    starti = 0
    L = np.zeros(len(Ks))
    for i in i_s:
        endi = starti + i
        hD = pairwise_distances(X=X[starti:endi, :], Y=X, metric='euclidean', n_jobs=n_jobs) ** 2
        lD = pairwise_distances(X=mappedX[starti:endi, :], Y=mappedX, metric='euclidean', n_jobs=n_jobs) ** 2
        starti += i
        ind1 = hD.argsort(axis=1)
        ind2 = lD.argsort(axis=1)
        for j in range(i):
            for k in range(len(Ks)):
                L[k] += len(np.intersect1d(ind1[j, 1:Ks[k] + 1], ind2[j, 1:Ks[k] + 1]))
    for k in range(len(Ks)):
        L[k] = L[k] / (n * Ks[k]) + Ks[k] / (1 - n)
    return L


def trustworthiness_large(X:np.array, mappedX:np.array, Ks:list=[1], block:int=20000, n_jobs:int=-1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n = X.shape[0]
    T = np.zeros(len(Ks))
    n_blocks = n // block
    for j in range(n_blocks):
        low_X = int(j * n / n_blocks)
        high_X = int((j + 1) * n / n_blocks)
        num_samples = X[low_X: high_X, :].shape[0]

        hD = pairwise_distances(X=X[low_X: high_X, :], Y=X, metric='euclidean', n_jobs=n_jobs) ** 2
        lD = pairwise_distances(X=mappedX[low_X: high_X, :], Y=mappedX, metric='euclidean', n_jobs=n_jobs) ** 2

        def worker(id, return_dict):
            low_pos = int(id * num_samples / n_jobs)
            high_pos = int((id + 1) * num_samples / n_jobs)
            if high_pos > num_samples:
                high_pos = num_samples
            ranks = np.zeros(np.max(Ks))
            innerT = np.zeros(len(Ks))
            ind1 = hD[low_pos:high_pos, :].argsort(axis=1)
            ind2 = lD[low_pos:high_pos, :].argsort(axis=1)
            for j in range(high_pos - low_pos):
                for m in range(np.max(Ks)):
                    ranks[m] = np.where(ind1[j, :] == ind2[j][m + 1])[0]
                ii = 0
                for k in Ks:
                    ranksi = ranks[:k] - k
                    innerT[ii] += np.sum(ranksi[ranksi > 0])
                    ii += 1
            return_dict[id] = innerT

        # ind1 = hD.argsort(axis=1)
        # ind2 = lD.argsort(axis=1)

        record = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for i in range(n_jobs):
            process = mp.Process(target=worker, args=(i, return_dict))
            process.start()
            record.append(process)
        for process in record:
            process.join()

        for PT in return_dict.values():
            T += PT

        for process in record:
            process.terminate()
            del process

    ii = 0
    for k in Ks:
        T[ii] = 1 - ((2 / (n * k * (2 * n - 3 * k - 1))) * T[ii])
        ii += 1
    return T
