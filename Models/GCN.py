from torch_geometric.nn import global_mean_pool
from Models.model_abstract import Model
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear
import torch


class GCN(Model):
    """
    Model base from:
        https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
    """

    def __init__(self, hidden_channels, dataset, seed=12345):
        """
        Basic classifier
        :param hidden_channels: number of hidden channel for each layer
        :param dataset: dataset used to train the model
        :param seed: random seed
        """
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(dataset.num_node_features,
                             hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Apply forward in the model
        :param x: features matrix
        :param edge_index: adjencency matrix
        :param batch: batch size
        :param  edge_weight: weights of the edges
        :return: output model
        """
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
