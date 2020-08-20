"""
"""

import os
import time
import json

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from data import load_knn_dataset
from .sampler import ClusterIter
from .gat import GAT
from .loss import TSNELoss
from utils.eval import evaluate, save_result
from utils.metrics import low_dimension_evaluate
from utils.plot import scatter


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(args, data_split_part=0, data_dir='./datasets', result_dir='./result'):
    # prepare data: graph, features, labels, manifold probability matrix
    nowstr = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
    result_dir = os.path.join(result_dir, args.dataset, nowstr)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'args.txt'), 'w') as fh:
        json.dump(args.__dict__, fh, indent=4)

    features, labels, P = load_knn_dataset(args=args, split_part=data_split_part, data_dir=data_dir)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    g = dgl.DGLGraph()
    g.from_scipy_sparse_matrix(P)

    g.ndata['features'] = features
    g.ndata['labels'] = labels
    # add edge weights
    coo_matrix = P.tocoo()
    g.edata['weight'] = torch.LongTensor(coo_matrix.data)   # why used to multiply 1000

    in_feats = features.shape[1]
    n_classes = 2
    n_edges = g.number_of_edges()
    g.readonly()
    
    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d""" %
            (n_edges, n_classes,
            features.shape[0],
            0))

    if args.gpu < 0:
        cuda = False
        print("Not GPU mode.")
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        P = P.cuda()
        print("use cuda:", args.gpu)

    # cluster_iterator = ClusterIter(g, args.psize, args.batch_size, 
    #                                seed_nid = None, 
    #                                use_pp = False)
    cluster_iterator = ClusterIter(dn=None, g=g, 
                                   psize=args.psize, 
                                   batch_size=args.batch_size, 
                                   seed_nid = None, 
                                   use_pp = False)

    # initialize model and loss function objects
    # create model
    heads = ([4] * args.n_layers) + [6]
    model = GAT(num_layers = args.n_layers,
                in_dim = in_feats,
                num_hidden = args.n_hidden,
                num_classes = n_classes,
                heads = heads,
                activation = F.elu,
                feat_drop = args.dropout,
                attn_drop = args.dropout,
                alpha = 0.2,
                residual = False)

    infer_model = GAT(num_layers = args.n_layers,
                in_dim = in_feats,
                num_hidden = args.n_hidden,
                num_classes = n_classes,
                heads = heads,
                activation = F.elu,
                feat_drop = 0.0,
                attn_drop = 0.0,
                alpha = 0.2,
                residual = False)
    if cuda:
        model.cuda()
        infer_model.cuda()
    loss_fcn = TSNELoss(cuda)
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # the train process for multiple epochs
    for epoch in range(1, args.n_epochs + 1):
        dur = []
        loss_values = []
        num_pos_edges = []
        num_neg_edges = []
        total_iter = 0
        for j, cluster in enumerate(cluster_iterator):
            t0 = time.time()

            ### add negative sampling edges
            pos_edges = cluster.parent_eid
            # print(f"======================================{j}======================================")
            # print("pos_cluster: ", cluster.number_of_nodes(), cluster.number_of_edges())
            pos_g, neg_g = next(dgl.contrib.sampling.EdgeSampler(g, 
                                                                batch_size=pos_edges.shape[0],
                                                                seed_edges=pos_edges,
                                                                # edge_weight=edge_probs,
                                                                # node_weight=node_probs,
                                                                negative_mode='tail',
                                                                # num_workers=1,
                                                                reset=True,
                                                                exclude_positive=True,
                                                                neg_sample_size=args.neg_sample).__iter__())
            cluster_nids = torch.unique(torch.cat([pos_g.parent_nid, neg_g.parent_nid]))
            # print("Edge of pos_g, neg_g: ", pos_g.parent_eid.shape, neg_g.parent_eid.shape)
            # print("Node of pos_g, neg_g, cluster_nids: ", pos_g.parent_nid.shape, neg_g.parent_nid.shape, cluster_nids.shape)
            cluster = g.subgraph(cluster_nids)
            # print("cluster: ", cluster.number_of_nodes(), cluster.number_of_edges())

            # sync with upper level training graph
            cluster.copy_from_parent()
            model.train()
            # forward
            logits = model(cluster)
            nids = cluster.parent_nid.numpy()
            cluster_P = P[nids[:, None], nids].todense()
            cluster_P = cluster_P / np.sum(cluster_P)
            cluster_P = np.maximum(cluster_P, 1e-12)
            cluster_P = torch.from_numpy(cluster_P)
            if cuda:
                cluster_P = cluster_P.cuda()

            if epoch <= 100:
                loss = loss_fcn(logits, cluster_P * 4)
            else:
                loss = loss_fcn(logits, cluster_P)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_values.append(loss_value)
            accs = [0]
            dur.append(time.time() - t0)
            num_pos_edges.append(pos_g.parent_eid.shape[0])
            num_neg_edges.append(neg_g.parent_eid.shape[0])

            total_iter += 1
            print("Iter {:05d} | Time(s) {:.4f} | Loss {:.4f} | Positive {:05d} | Negtive {:05d}".
                                        format(total_iter, dur[-1], loss_value, num_pos_edges[-1], num_neg_edges[-1]))
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {} | "
            "ETputs(KTEPS) {:.2f}".format(epoch, np.sum(dur), np.mean(loss_values),
                                        accs[-1], (np.sum(num_pos_edges) + np.sum(num_neg_edges)) / np.mean(dur) / 1000))
        # return
        if epoch % args.val_every == 0:
            temp_result_dir = os.path.join(result_dir, f'epoch_{epoch}')
            os.makedirs(temp_result_dir, exist_ok=True)
            eval_metric_path = os.path.join(temp_result_dir, 'metrics.txt')
            eval_data_path = os.path.join(temp_result_dir, 'train_embedding.npz')
            eval_scatter_path = os.path.join(temp_result_dir, 'plot_result.png')
            evaluate(model, infer_model, g, features.numpy(), labels.numpy(), 
                    ks=[1, 5, 10],
                    eval_metric_path = eval_metric_path,
                    eval_data_path = None,
                    eval_scatter_path = eval_scatter_path)

    # evaluation and save results
    eval_metric_path = os.path.join(result_dir, 'metrics.txt')
    eval_data_path = os.path.join(result_dir, 'train_embedding.npz')
    eval_scatter_path = os.path.join(result_dir, 'plot_result.png')
    evaluate(model, infer_model, g, features.numpy(), labels.numpy(),
             ks=[1, 5, 10],
             eval_metric_path = eval_metric_path,
             eval_data_path = eval_data_path,
             eval_scatter_path = eval_scatter_path)

    # save model
    save_model_path = os.path.join(result_dir, 'model.pt')
    torch.save(model.state_dict(), save_model_path)


def test(args, result_dir, data_split_part=1):
    # load dataset
    g, features, labels, P = load_knn_dataset(args=args, split_part=data_split_part)
    in_feats = features.shape[1]
    n_classes = 2
    n_edges = g.number_of_edges()
    g.readonly()
    # load model
    model_path = os.path.join(result_dir, 'model.pt')
    heads = ([4] * args.n_layers) + [6]
    infer_model = GAT(num_layers = args.n_layers,
                in_dim = in_feats,
                num_hidden = args.n_hidden,
                num_classes = n_classes,
                heads = heads,
                activation = F.elu,
                feat_drop = 0.0,
                attn_drop = 0.0,
                alpha = 0.2,
                residual = False)
    infer_model.load_state_dict(torch.load(model_path))

    eval_metric_path = os.path.join(result_dir, 'test_metrics.txt')
    eval_scatter_path = os.path.join(result_dir, 'test_plot_result.png')
    eval_data_path = os.path.join(result_dir, 'test_embedding.npz')

    infer_model.eval()
    with torch.no_grad():
        embeddings = infer_model(g)
        embeddings = embeddings.cpu().numpy()
        features = features.numpy()
        labels = labels.numpy()
        ks = [1, 5, 10, 20, 30, 40, 50]
        T, L = low_dimension_evaluate(features, embeddings, ks=ks)
        with open(eval_metric_path, 'w') as fh:
            fh.write(','.join([str(k) for k in ks]) + '\n')
            fh.write(','.join([str(t) for t in T]) + '\n')
            fh.write(','.join([str(l) for l in L]) + '\n')
        if eval_scatter_path: 
            scatter(embeddings, labels, eval_scatter_path)
        if eval_data_path:
            save_result(embeddings, filepath=eval_data_path)


def predict(args, train_split_part, predict_split_part, data_dir='./datasets'):
    features, labels, P = load_knn_dataset(args=args, split_part=train_split_part, data_dir=data_dir)
    train_g = dgl.DGLGraph()
    train_g.from_scipy_sparse_matrix(P)
    print(train_g.number_of_nodes())
    train_g.ndata['features'] = torch.FloatTensor(features)
    train_g.ndata['labels'] = torch.LongTensor(labels)

    predict_features, predict_labels, _ = load_knn_dataset(args=args, split_part=predict_split_part, data_dir=data_dir)
    predict_features = torch.FloatTensor(predict_features)
    predict_labels = torch.LongTensor(predict_labels)

    from .predict import build_knn_from_training_data
    predict_g = build_knn_from_training_data(predict_features[0:2], predict_labels[0:2], 'annoy_index', args.k, train_g, n_layer=1)
    print(predict_g.number_of_nodes())
    print(predict_g.parent_nid)
    print(predict_g.ndata)
    # print(predict_g.ndata['features'])
    
