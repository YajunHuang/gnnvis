from knn import KNNIndex


def build_knn_from_training_data(features, labels, index_file, k, train_g, n_layer):
    unseen_index_start = train_g.number_of_nodes()
    knn_index = KNNIndex(vec_len=features.shape[1], index_file=index_file)
    new_edges_src = list()
    new_edges_dst = list()
    train_sub_g_nodes = set()

    # search unseen data knn from training data
    for i in range(features.shape[0]):
        nns = knn_index.get_nns_by_vector(v=features[i],
                                          n=k,
                                          n_propagation=3)
        for n in nns:
            new_edges_src.append(unseen_index_start + i)
            new_edges_dst.append(n)
            new_edges_src.append(n)
            new_edges_dst.append(unseen_index_start + i)
            train_sub_g_nodes.add(n)
    # add unseen data knn to training data knn
    train_g.add_nodes(features.shape[0], {'features': features, 'labels': labels})
    train_g.add_edges(new_edges_src, new_edges_dst)

    # search n_layer hop neighbors for unseen data nodes
    expand_seeds = train_sub_g_nodes
    for i in range(n_layer - 1):
        expand_nodes = set()
        for seed in expand_seeds:
            expand_nodes = expand_nodes | set(train_g.successors(seed).numpy())
        expand_seeds = expand_nodes - expand_seeds
        train_sub_g_nodes = train_sub_g_nodes | expand_nodes
    train_sub_g_nodes = train_sub_g_nodes | set(new_edges_src)
    test_knn_g = train_g.subgraph(list(train_sub_g_nodes))

    nids = test_knn_g.parent_nid
    test_knn_g.ndata['features'] = train_g.ndata['features'][nids[:, None], :]
    test_knn_g.ndata['labels'] = train_g.ndata['labels'][nids]
    # return test_ids, test_knn_g, test_knn_features
    return test_knn_g
