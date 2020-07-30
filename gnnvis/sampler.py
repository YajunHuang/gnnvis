import os
import random
from time import time

import metis
import numpy as np
import dgl.function as fn
import torch


class ClusterIter(object):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, g, psize, batch_size, seed_nid=None, use_pp=True, cache_dir=None):
        """Initialize the sampler.

        Paramters
        ---------
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        use_pp: bool
            Whether to use precompute of AX
        cache_dir : str
            The graph partition result cache directory.
        """
        self.use_pp = use_pp
        if seed_nid:
            self.g = g.subgraph(seed_nid)
            self.g.copy_from_parent()
        else:
            self.g = g
        
        # precalc the aggregated features from training graph only
        if use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = psize
        self.batch_size = batch_size
        # cache the partitions of known datasets&partition number
        if cache_dir:
            fn = os.path.join(cache_dir, 'cluster_partitions_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs(cache_dir, exist_ok=True)
                self.par_li = get_partition_list(self.g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(self.g, psize)
        self.max = int((psize) // batch_size)
        random.shuffle(self.par_li)
        self.get_fn = get_subgraph

    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        features = g.ndata['features']
        print("features shape, ", features.shape)
        with torch.no_grad():
            g.update_all(fn.copy_src(src='features', out='m'),
                         fn.sum(msg='m', out='features'),
                         None)
            pre_feats = g.ndata['features'] * norm
            # use graphsage embedding aggregation style
            g.ndata['features'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['features'].device)
        return norm

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.get_fn(self.g, self.par_li, self.n,
                                 self.psize, self.batch_size)
            self.n += 1
            return result
        else:
            random.shuffle(self.par_li)
            raise StopIteration


def get_partition_list(g, psize):
    """ Partition the graph into psize parts using Metis algorithm.

    Parameters:
    -----------
    g: DGLGraph
        The graph used to partition
    psize: int
        The partition number 

    Return:
    -------
    al: 2D array
        The graph partition result
    """
    tmp_time = time()
    ng = g.to_networkx(edge_attrs=['weight'])
    ng.graph['edge_weight_attr'] = 'weight'
    print("getting adj using time {:.4f} seconds".format(time() - tmp_time))
    print("run metis with partition size {}".format(psize))
    _, nd_group = metis.part_graph(ng, psize)
    print("metis finished in {:.4f} seconds.".format(time() - tmp_time))
    print("train group {}".format(len(nd_group)))
    al = arg_list(nd_group)
    return al


def get_subgraph(g, par_arr, i, psize, batch_size):
    """ Get subgraph of g for the i-th batch of graph partition.

    Parameters:
    -----------
    g: GDLGraph
    par_arr: 2D array
        the partition arrays of the graph, each row is the nodes of a subgraph.
    i: int
        the batch index.
    psize: int
        The partition number
    batch_size: int
        The number of partitions in one batch

    Return:
    -------
    subg: GDLGraph
        the subgraph of g
    """
    par_batch_ind_arr = [par_arr[s] for s in range(
        i * batch_size, (i + 1) * batch_size) if s < psize]
    subg = g.subgraph(np.concatenate(
        par_batch_ind_arr).reshape(-1).astype(np.int64))
    return subg


def arg_list(labels):
    """ Partition labels by label and get their indices

    Parameters:
    -----------
    labels: list
        The 1D array of labels

    Return:
    -------
    label_index_lists: 2D array
        The index lists of different labels

    """
    hist, indexes, inverse, counts = np.unique(
        labels, return_index=True, return_counts=True, return_inverse=True)
    label_index_lists = []
    for h in hist:
        label_index_lists.append(np.argwhere(inverse == h))
    return label_index_lists
