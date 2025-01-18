import os.path as osp
import numpy as np
from cvxopt import matrix, solvers
from sklearn.kernel_approximation import RBFSampler
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from sklearn.metrics import f1_score
import random
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def evaluate(data, model):
    model.eval()
    output = model(data.x, data.edge_index)
    
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output, data.y)
    pred = output.max(dim=1)[1]
    
    correct = pred.eq(data.y).sum().item()
    acc = correct * 1.0 / len(data.y)

    pred = pred.cpu().numpy()
    gt = data.y.cpu().numpy()
    macro_f1 = f1_score(gt, pred, average='macro')
    micro_f1 = f1_score(gt, pred, average='micro')

    # 计算并打印每一类的准确率
    class_accuracies = {}
    for cls in np.unique(gt):
        class_mask = (gt == cls)
        class_correct = (pred[class_mask] == gt[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc = class_correct * 1.0 / class_total
        class_accuracies[cls] = class_acc
        #print(f'Class {cls}: Accuracy = {class_acc:.4f}')

    return acc, macro_f1, micro_f1, loss,class_accuracies


def guassian_kernel(source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):

    n_samples = int(source.size()[0]) + int(target.size()[0])  
    total = torch.cat([source, target], dim=0) 
    total0 = total.unsqueeze(0).expand(int(total.size(0)), 
                                       int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), 
                                       int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

# 计算样本之间的距离
def pairwise_distances(x, y=None):
    if y is None:
        y = x
    dists = torch.cdist(x, y, p=2)
    return dists

def approximate_kernel(X, Xtest, gamma=1.0, n_components=100):
    rbf_sampler = RBFSampler(gamma=gamma, n_components=n_components)
    X_features = rbf_sampler.fit_transform(X.detach().cpu().numpy())
    Xtest_features = rbf_sampler.transform(Xtest.detach().cpu().numpy())
    return torch.tensor(X_features).to(X.device), torch.tensor(Xtest_features).to(Xtest.device)


def KMM(X, Xtest, _A=None, gamma=1.0, n_components=100):
    device = X.device  # 获取输入张量所在的设备

    # 使用随机特征近似
    X_features, Xtest_features = approximate_kernel(X, Xtest, gamma=gamma, n_components=n_components)

    H = torch.matmul(X_features, X_features.T)
    f = torch.matmul(X_features, Xtest_features.T)
    z = torch.matmul(Xtest_features, Xtest_features.T)

    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0], 1), dtype=torch.float64).to(device))
    eps = 10
    G = - np.eye(nsamples, dtype=np.float64)  # 确保类型为 float64
    h = - 0.2 * np.ones((nsamples, 1), dtype=np.float64)  # 确保类型为 float64
    A, b = None, None

    if _A is not None:
        A = matrix(_A.astype(np.float64))
        b = matrix(np.ones((_A.shape[0], 1), dtype=np.float64) * 20)

    try:
        solvers.options['show_progress'] = False
        sol = solvers.qp(matrix(H.detach().cpu().numpy().astype(np.float64)),
                         matrix(f.detach().cpu().numpy().astype(np.float64)),
                         matrix(G),
                         matrix(h),
                         A,
                         b)
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!Optimization failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: {e}")
        return torch.ones(nsamples, 1, dtype=torch.float64).to(device), MMD_dist  # 返回全1的权重张量

    return torch.tensor(np.array(sol['x']), device=device, dtype=torch.float64), MMD_dist.item()


# 计算样本权重
def compute_sample_weights(source_data, target_data, model, args):
    device = next(model.parameters()).device  # 获取模型所在的设备
    source_feature = model.feat_bottleneck(source_data.x.to(device), source_data.edge_index.to(device), args.source_pnum)
    target_feature = model.feat_bottleneck(target_data.x.to(device), target_data.edge_index.to(device), args.target_pnum)
    weights, _ = KMM(source_feature, target_feature)
    return weights

def MMD(source_feat, target_feat, sampling_num = 1000, times = 5):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)

    mmd = mmd / times
    return mmd

def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5
            , fix_sigma=None):
    kernels = guassian_kernel(source_feat, 
                              target_feat,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    
    batch_size = min(int(source_feat.size()[0]), int(target_feat.size()[0]))  
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    to_number = int
    if torch.is_floating_point(torch.empty(0, dtype=dtype)):
        to_number = float

    src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src).to(dtype).squeeze()
    return src
def read_txt_array(path, sep=None, start=0, end=None, noise_ratio=0.0, dtype=None, device=None):
    src = []
    with open(path, 'r') as f:
        for line in f:
            if random.random() >= noise_ratio:
                src.append(line.strip())  # 去除每行末尾的换行符和空格
    return parse_txt_array(src, sep, start, end, dtype, device)
class CitationDataset(InMemoryDataset):
    seed(42)
    def __init__(self,
                 root,
                 name,
                 noise_ratio = 0.0,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        self.noise_ratio = noise_ratio
        super(CitationDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])  # 加载之前数据
        #self.data, self.slices = self.process()  # 每次重新处理数据，不加载已处理数据
    
    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',',  noise_ratio=self.noise_ratio, dtype=torch.long).t()
        print('ratio: ',self.noise_ratio, 'edge number:', len(edge_index[0]))

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
           data = self.pre_transform(data)

        data_list.append(data)

        #data, slices = self.collate([data])
        data, slices = self.collate(data_list)

        return data, slices

        #torch.save((data, slices), self.processed_paths[0])


class TwitchDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        super(TwitchDataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ["edges.csv, features.json, target.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def load_twitch(self, lang):
        assert lang in ('DE', 'EN', 'FR'), 'Invalid dataset'
        filepath = self.raw_dir
        label = []
        node_ids = []
        src = []
        targ = []
        uniq_ids = set()
        with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node_id = int(row[5])
                # handle FR case of non-unique rows
                if node_id not in uniq_ids:
                    uniq_ids.add(node_id)
                    label.append(int(row[2]=="True"))
                    node_ids.append(int(row[5]))

        node_ids = np.array(node_ids, dtype=np.int)

        with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                src.append(int(row[0]))
                targ.append(int(row[1]))
        
        with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
            j = json.load(f)

        src = np.array(src)
        targ = np.array(targ)
        label = np.array(label)

        inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
        reorder_node_ids = np.zeros_like(node_ids)
        for i in range(label.shape[0]):
            reorder_node_ids[i] = inv_node_ids[i]
    
        n = label.shape[0]
        A = scipy.sparse.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape=(n,n))
        features = np.zeros((n,3170))
        for node, feats in j.items():
            if int(node) >= n:
                continue
            features[int(node), np.array(feats, dtype=int)] = 1
        new_label = label[reorder_node_ids]
        label = new_label
    
        return A, label, features

    def process(self):
        A, label, features = self.load_twitch(self.name)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        features = np.array(features)
        x = torch.from_numpy(features).to(torch.float)
        y = torch.from_numpy(label).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])
        
        
class Writer(object):
    def __init__(self, path):
        self.writer = SummaryWriter(path)
        
    def scalar_logger(self, tag, value, step):
        """Log a scalar variable."""
        # if self.local_rank == 0:
        self.writer.add_scalar(tag, value, step)
        
    def scalars_logger(self, tag, value, step):
        """Log a scalar variable."""
        # if self.local_rank == 0:
        self.writer.add_scalars(tag, value, step)

    def image_logger(self, tag, images, step):
        """Log a list of images."""
        # if self.local_rank == 0:
        self.writer.add_image(tag, images, step)

    def histo_logger(self, tag, values, step):
        """Log a histogram of the tensor of values."""
        # if self.local_rank == 0:
        self.writer.add_histogram(tag, values, step, bins='auto')