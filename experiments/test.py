import argparse

from gnnvis.gnnvis import *
from data import register_data_args
from data.mnist import MNISTDataset


def split_dataset(args):
    split_rates = [5, 3, 2]
    mnist = MNISTDataset(data_dir = args.data_dir, n_samples=args.dsize, k=args.k, split_rates=split_rates)
    mnist.make_sparse_graph_npz()


def main(args):
    split_dataset(args)
    # train(args)
    test(args, result_dir=args.res_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnnvis')
    register_data_args(parser)

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--dsize", type=int, default=6000,
                        help="number of data samples")
    parser.add_argument("--data_dir", type=str, default='datasets',
                        help="the dataset directory")
    parser.add_argument("--res_dir", type=str, default='datasets',
                        help="the test result directory")
    parser.add_argument("--k", type=int, default=30,
                        help="k nearest neighbors")

    args = parser.parse_args()

    print(args)
    main(args)
