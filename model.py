import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from layers import GCNConv
from torch_geometric.nn import SGConv

class Model(torch.nn.Module):
    def __init__(self, args,use_mlp=True):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        #self.cls = GCNConv(self.nhid, self.num_classes)
        self.cls = torch.nn.Linear(self.nhid, self.num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.nhid, self.nhid),
            torch.nn.Linear(self.nhid, self.nhid),
            torch.nn.BatchNorm1d(self.nhid),
            torch.nn.ReLU(),
        )

    def forward(self, x, edge_index, conv_time=30):
        x = self.feat_bottleneck(x, edge_index, conv_time)
        x = self.feat_classifier(x)

        return x

    def feat_bottleneck(self, x, edge_index, conv_time=30):
        x = self.conv1(x, edge_index, conv_time)
        x = F.relu(x)
        return x


    def feat_classifier(self, x):
        x = self.cls(x)
        return x


class GCNModel(torch.nn.Module):
    def __init__(self, args,adj_matrix):
        super(GCNModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
        self.convs.append(GCNConv(self.hidden_dim, self.num_classes))

    def forward(self, x, edge_index,conv_time=30):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index,  conv_time=30):
        x = self.conv1(x, edge_index, conv_time)
        x = F.relu(x)
        return x
    #
class GATModel(torch.nn.Module):
    def __init__(self, args, adj_matrix):
        super(GATModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_layers = args.num_layers
        self.num_heads = 1


        self.conv1 = GATConv(self.num_features, self.hidden_dim, heads=1, concat=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers - 2):
            self.convs.append(GATConv(self.hidden_dim * self.num_heads, self.hidden_dim, heads=self.num_heads, concat=True))
        self.convs.append(GATConv(self.hidden_dim * self.num_heads, self.num_classes, heads=1, concat=False))

    def forward(self, x, edge_index, conv_time=30):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index, conv_time=30):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return x

class SGCModel(torch.nn.Module):
    def __init__(self, args, adj_matrix):
        super(SGCModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio

        self.conv = SGConv(self.num_features, self.num_classes, K=args.num_layers)

    def forward(self, x, edge_index, conv_time=30):
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)

    def feat_bottleneck(self, x, edge_index, conv_time=30):
        x = self.conv(x, edge_index)
        return x


class Robust_Model(torch.nn.Module):
    def __init__(self, args,adj_matrix):
        super(Robust_Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.num_source = args.num_source
        self.num_target = args.num_target
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.cls = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, x, edge_index, conv_time=30):
        x = self.feat_bottleneck(x, edge_index, conv_time)
        x = self.feat_classifier(x)
        return x

    def feat_bottleneck(self, x, edge_index, conv_time=30):
        x = self.conv1(x, edge_index, conv_time)
        x = F.relu(x)
        return x

    def feat_classifier(self, x):
        x = self.cls(x)
        return x


