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
    train(args)
    # test(args, result_dir='/Users/yale/Projects/research/gnnvis/result/mnist/2020-04-22-153247')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gnnvis')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--val-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use percomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--note", type=str, default='none',
                        help="note for log dir")
    parser.add_argument("--dsize", type=int, default=6000,
                        help="number of data samples")
    parser.add_argument("--data_dir", type=str, default='datasets',
                        help="the dataset directory")
    parser.add_argument("--k", type=int, default=30,
                        help="k nearest neighbors")

    args = parser.parse_args()

    print(args)
    main(args)