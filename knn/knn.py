import os
import numpy as np
import os
import networkx as nx
import pandas as pd
import scipy.sparse as sp

from annoy import AnnoyIndex
from .sparse import SparseGraph

import KNN as knn


__all__ = ['construct_sparse_knn_graph', 'KNNIndex']


class KNNIndex(object):
  annoy = None
  vec_len = -1
  metric = 'euclidean'
  is_loaded = False

  def __init__(self, vec_len, metric='euclidean', index_file=None):
    self.vec_len = vec_len
    self.metric = metric
    self.annoy = AnnoyIndex(self.vec_len, self.metric)
    if index_file:
      self.load(index_file)

  def get_nns_by_item(self, i, n, search_k=-1, include_distances=False):
    if self.is_loaded:
      return self.annoy.get_nns_by_item(i, n, search_k, include_distances)
    else:
      raise RuntimeError("Annoy index file is not loaded!")

  def get_nns_by_vector(self, v, n, search_k=-1, include_distances=False, n_propagation=0):
    if self.is_loaded:
      return self.annoy.get_nns_by_vector(v, n, search_k, include_distances)
    else:
      raise RuntimeError("Annoy index file is not loaded!")

  def load(self, index_file):
    self.annoy.load(index_file)
    self.is_loaded = True
  
  def unload(self):
    self.annoy.unload()
    self.is_loaded = False


# def construct_knn(feature_in_lists:list, perplexity:float=30.0, P_knn:int=10):
#     import KNN as knn
#     P_knn = int(perplexity * 3)
#     knn.n_neighbors(P_knn)
#     knn.load_data(feature_in_lists)
#     dir = os.getcwd()
#     knn.construct_knn(10, 3, perplexity, "{}/data/MNIST/mnist_knn.txt".format(dir))
#     edgelist_df = pd.read_csv("{}/data/MNIST/mnist_knn.txt".format(dir), sep=" ", header=None, dtype={0:'int', 1:'int', 2:'float'})
#     edgelist_df.columns = ["src", "tar", "weight"]
#     return edgelist_df


def construct_sparse_knn_graph(features: np.ndarray, k: int = 30):
    n_tree = 10
    n_propagation = 3
    perplexity = 100
    n_thread = 4
    knn_result = knn.build_knn_index(features.astype(np.float), k, 
                                     n_tree, n_propagation, perplexity, n_thread)
    srcs = knn_result[0]
    tars = knn_result[1]
    weights = knn_result[2]
    return sp.coo_matrix((weights, (srcs, tars))).tocsr()