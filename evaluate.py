import torch
from utils import MMD, evaluate, CitationDataset, TwitchDataset
import random
import argparse
import numpy as np
import os.path as osp
from model import Model
def seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=3, help='numer of layer')
    parser.add_argument('--nhid', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience for early stopping')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='specify cuda devices')
    parser.add_argument('--run_times', type=int, default=1,
                        help='run times')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='maximum number of epochs')
    parser.add_argument('--source', type=str, default='DBLPv7',
                        help='source domain data')  # DBLPv7
    parser.add_argument('--target', type=str, default='ACMv9',
                        help='target domain data')
    parser.add_argument('--weight', type=float, default=5,
                        help='trade-off parameter')
    parser.add_argument('--source_pnum', type=int, default=1,
                        help='the number of propagation layers on the source graph')
    parser.add_argument('--target_pnum', type=int, default=30,
                        help='the number of propagation layers on the target graph')
    args = parser.parse_args()
    seed(args.seed)
    print(args)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data',
                    args.target)
    for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98]:
        target_dataset = CitationDataset(path, args.target, noise_ratio=i)
        target_data = target_dataset[0].to(args.device)
        args.num_classes = len(np.unique(target_dataset[0].y.numpy()))
        args.num_features = target_data.x.size(1)
        args.save_path = './'

        model = Model(args).to(args.device)
        best_model = 255
        model.load_state_dict(torch.load(args.save_path+'{}.pth'.format(best_model)))

        _, macro_f1, micro_f1, test_loss, _ = evaluate(target_data, model)

        print('{:}, {} -> {}  , macro_f1 = {:.6f}, \
            micro_f1 = {:.6f}'.format(i, args.source, args.target, macro_f1, micro_f1))
