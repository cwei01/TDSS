import argparse
import itertools
import torch.nn.utils as utils
import os
import os.path as osp
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import logging
import torch
import torch.nn.functional as F
from model import Robust_Model, GCNModel,GATModel,SGCModel
from utils import MMD, evaluate, CitationDataset, TwitchDataset
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from torch_geometric.utils import k_hop_subgraph


def seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.manual_seed(args.seed)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def adj_to_edge_index(adj_matrix):
    # Get indices of the non-zero elements
    edge_index = torch.nonzero(adj_matrix).t()
    return edge_index

def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # Create a zero matrix of size num_nodes x num_nodes.
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    return adjacency_matrix


def compute_laplacian_loss(features, edge_index):
    edge_weight = torch.ones(edge_index.size(1), device=features.device)
    row, col = edge_index
    deg = torch.zeros(features.size(0), device=features.device)
    deg = deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    laplacian_loss = (features[row] - features[col]).pow(2).sum(dim=1) * normalized_edge_weight
    return laplacian_loss.sum()


def random_walk_neighborhood(edge_index, num_nodes, num_walks, walk_length):
    neighbors = [[] for _ in range(num_nodes)]

    for node in range(num_nodes):
        for _ in range(num_walks):
            current_node = node
            for _ in range(walk_length):
                # Add the current node to the neighbor list of the starting node
                neighbors[node].append(current_node)
                neighbors[current_node].append(node)

                # Get all edges of the current node
                edges = edge_index[:, edge_index[0] == current_node]
                # Stop the random walk if there are no more neighbors
                if edges.size(1) == 0:
                    break
                # Randomly select a neighbor as the next node
                next_node = edges[1][torch.randint(edges.size(1), (1,)).item()]

                # Update the current node to the next node
                current_node = next_node

    # Create a new edge list
    new_edge_index = []
    for node, neigh in enumerate(neighbors):
        for neighbor in set(neigh):
            new_edge_index.append([node, neighbor])

    # Convert to Tensor
    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()

    return new_edge_index
def compute_k_hop_neighbors(edge_index, num_nodes, k, proportion):
    edge_index_k_hop = edge_index.clone()

    for _ in range(k - 1):
        new_edge_index_k_hop = []
        for node in range(num_nodes):
            # Get the 1-hop neighbors of the node
            _, edge_index_neigh, _, _ = k_hop_subgraph(node, 1, edge_index_k_hop, relabel_nodes=False)
            num_neighbors = edge_index_neigh.size(1)

            # Select neighbor nodes according to the proportion
            num_sampled_neighbors = int(proportion * num_neighbors)
            if num_sampled_neighbors < num_neighbors:
                sampled_indices = torch.randperm(num_neighbors)[:num_sampled_neighbors]
                edge_index_neigh = edge_index_neigh[:, sampled_indices]

            new_edge_index_k_hop.append(edge_index_neigh)

        # Merge all edges
        edge_index_k_hop = torch.cat(new_edge_index_k_hop, dim=1).unique(dim=1)

    return edge_index_k_hop

def compute_k_hop_laplacian_loss_old(features, edge_index_k_hop):
    # Compute the Laplacian constraint loss
    edge_weight = torch.ones(edge_index_k_hop.size(1), device=features.device)
    row, col = edge_index_k_hop.to(features.device)
    deg = torch.zeros(features.size(0), device=features.device)
    deg = deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    normalized_edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    laplacian_loss = (features[row] - features[col]).pow(2).sum(dim=1) * normalized_edge_weight
    return laplacian_loss.sum()

def compute_k_hop_laplacian_loss(features, edge_index_k_hop):
    # Compute the Laplacian constraint loss
    edge_weight = torch.ones(edge_index_k_hop.size(1), device=features.device)
    row, col = edge_index_k_hop.to(features.device)

    # Compute the degree of each node
    deg = torch.zeros(features.size(0), device=features.device)
    deg = deg.scatter_add_(0, row, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    # Compute the normalized feature differences
    normalized_features_row = features[row] * deg_inv_sqrt[row].view(-1, 1)
    normalized_features_col = features[col] * deg_inv_sqrt[col].view(-1, 1)
    feature_diff = normalized_features_row - normalized_features_col

    # Compute the loss
    laplacian_loss = (feature_diff).pow(2).sum(dim=1) * edge_weight
    return laplacian_loss.sum() / 2.

def train(args, source_data, target_data,file_path,run_times):
    num_source = source_data.x.size(0)
    adj_source = edge_index_to_adjacency_matrix(source_data.edge_index, num_source).to(args.device)
    #
    num_target = target_data.x.size(0)
    adj_target = edge_index_to_adjacency_matrix(target_data.edge_index, num_target).to(args.device)

    patience_cnt = 0
    best_f1 = 0
    f1_values = []
    model = Robust_Model(args, adj_source).to(args.device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 在训练开始时计算k-hop邻域
    edge_index_k_hop = compute_k_hop_neighbors(target_data.edge_index, num_target,args.k,args.ratio)
    edge_index_random_walk = random_walk_neighborhood(target_data.edge_index, num_target, args.rw_num, args.k)
    for epoch in range(args.epochs):
        #logger.info("********* Epoch: {:04d} *********".format(epoch + 1))
        #logger.info("*** Running training***")
        model.train()
        optimizer.zero_grad()
        # Source Domain Cross-Entropy Loss
        output = model(source_data.x, source_data.edge_index, args.source_pnum)
        train_loss = F.nll_loss(F.log_softmax(output, dim=1), source_data.y)
        loss = train_loss

        source_feature = model.feat_bottleneck(source_data.x, source_data.edge_index, args.source_pnum)
        target_feature = model.feat_bottleneck(target_data.x, target_data.edge_index, args.target_pnum)
        mmd_loss = MMD(source_feature,target_feature)

        # Laplacian Regularization Loss
        laplacian_loss_target = compute_k_hop_laplacian_loss(target_feature, edge_index_random_walk)


        loss = loss +  args.mmd_weight * mmd_loss+\
               args.laplacian_target_weight*laplacian_loss_target
               #args.laplacian_source_weight*laplacian_loss_source
        loss.backward()
        optimizer.step()

        with torch.no_grad():
               # logger.info("*** Running evaluation ***")
                acc, _, _, _, _ = evaluate(source_data, model)
                _, macro_f1, micro_f1, test_loss, calss_accuracies = evaluate(target_data, model)
                if (epoch + 1) % 50 == 0:
                    print('Epoch: {:04d}'.format(epoch + 1), 'train_loss: {:.4f}'.format(loss),'test_loss: {:.4f}'.format(test_loss),
                          'macro_f1: {:.4f}'.format(macro_f1), 'micro_f1: {:.4f}'.format(micro_f1))
                    print('Epoch: {:04d}'.format(epoch + 1), 'train_loss: {:.4f}'.format(loss), 'test_loss: {:.4f}'.format(test_loss),
                          'train_acc: {:.4f}'.format(acc), 'macro_f1: {:.4f}'.format(macro_f1),'micro_f1: {:.4f}'.format(micro_f1), file=file_path)
                    file_path.flush()
        f1_values.append(macro_f1)
        if f1_values[-1] > best_f1:
            torch.save(model.state_dict(), args.save_path +'{}.pth'.format(run_times))
            best_f1 = f1_values[-1]
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt == args.patience:
            break

    return model

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        #### trade-off parameter ####
        parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')
        parser.add_argument('--mmd_weight', type=float, default=7,help='trade-off parameter')
        parser.add_argument('--laplacian_target_weight', type=float, default=2e-4, help='trade-off parameter')
        parser.add_argument('--laplacian_source_weight', type=float, default=0, help='trade-off parameter')
        parser.add_argument('--k', type=int, default=1, help='k-hop')
        parser.add_argument('--rw_num', type=int, default=4, help='rw_num')
        parser.add_argument('--ratio', type=float, default=1, help='sample ratio')
        parser.add_argument('--output_dir', type=str, default='output/DC',help='output_dir')
        parser.add_argument('--source_pnum', type=int, default=0,help='the number of propagation layers on the source graph')
        parser.add_argument('--target_pnum', type=int, default=35, help='the number of propagation layers on the target graph')
        parser.add_argument('--source', type=str, default='Citationv1', help='source domain data') #DBLPv7 #DBLPv7DBLPv7DBLPv7ACMv9 Citationv1
        parser.add_argument('--target', type=str, default='ACMv9', help='target domain data')
        parser.add_argument('--run_times', type=int, default=3,help='run times')
        parser.add_argument('--epochs', type=int, default=800, help='maximum number of epochs')

        parser.add_argument('--seed', type=int, default=200,help='random seed')
        parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
        parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
        parser.add_argument('--num_layers',type=int,default=4,help='numer of layer')
        parser.add_argument('--nhid', type=int, default=128, help='hidden size')
        parser.add_argument('--patience', type=int, default=700, help='patience for early stopping')
        parser.add_argument('--weight', type=float, default=5,help='trade-off parameter')
        args = parser.parse_args()
        seed(args.seed)
        print(args)


        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        log_path = os.path.join(args.output_dir, 'performance.txt')
        parameter_path = os.path.join(args.output_dir, 'parameter.txt')

        f = open(parameter_path, "w")
        for arg in sorted(vars(args)):
            print("{}: {}".format(arg, getattr(args, arg)), file=f)
        f.close()
        if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', args.source)
            source_dataset = CitationDataset(path, args.source,noise_ratio=0.0)
        if args.source in {'EN', 'DE'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', args.source)
            source_dataset = TwitchDataset(path, args.source)
        if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data', args.target)
            target_dataset = CitationDataset(path, args.target,noise_ratio=0.0)
            noise_dataset = CitationDataset(path, args.target,noise_ratio=0.5)
        if args.target in {'EN', 'DE'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), '../', 'data',args.target)
            target_dataset = TwitchDataset(path, args.target)
        source_data = source_dataset[0].to(args.device)
        #print(len(source_dataset[0][0]))
        #print("f")
        target_data = target_dataset[0].to(args.device)
        noise_data = noise_dataset[0].to(args.device)

        args.num_classes = len(np.unique(source_dataset[0].y.numpy()))
        args.num_features = source_data.x.size(1)
        args.save_path = args.output_dir +'/'
        args.num_source = source_data.x.size(0)
        args.num_target = target_data.x.size(0)
        # Run Experiment
        macro_f1_dict = []
        micro_f1_dict = []
        all_dict = []
        time_dict = []
        file_path = open(log_path, "a")
        for i in range(args.run_times):
                model= train(args, source_data, target_data,file_path,i+1)
                model.load_state_dict(torch.load(args.save_path+'{}.pth'.format(i+1)))
                acc, _, _, _ ,_= evaluate(source_data, model)
                _, macro_f1, micro_f1, test_loss,_ = evaluate(target_data, model)

                result_str = '***{} -> {}***  source acc = {:.4f}, macro_f1 = {:.4f}, micro_f1 = {:.4f}'.format(args.source, args.target, acc, macro_f1, micro_f1)
                print(result_str)
                print(result_str, file=file_path)
                file_path.flush()

                macro_f1_dict.append(macro_f1)
                micro_f1_dict.append(micro_f1)

        macro_f1_dict_print = [float('{:.4f}'.format(i)) for i in macro_f1_dict]
        micro_f1_dict_print = [float('{:.4f}'.format(i)) for i in micro_f1_dict]
        all_dict_print = [float('{:.6f}'.format(i)) for i in all_dict]

        macro_result_str = '***mAcro: {} mean {:.4f} std {:.4f}***'.format(macro_f1_dict_print, np.mean(macro_f1_dict), np.std(macro_f1_dict))
        micro_result_str = '***mIcro: {} mean {:.4f} std {:.4f}***'.format(micro_f1_dict_print, np.mean(micro_f1_dict), np.std(micro_f1_dict))

        print(macro_result_str)
        print(macro_result_str, file=file_path)
        file_path.flush()
        print(micro_result_str)
        print(micro_result_str, file=file_path)
        file_path.flush()
        f.close()
        logger.info("****Writing predictions to*** %s" % (args.output_dir))



