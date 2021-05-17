from torch_geometric.utils import dense_to_sparse
from torch_geometric import data
import numpy as np
import random
import torch


class SyntheticGenerator:

    def __init__(self):

        self.num_nodes = 9  # all graphs have exactly 9 nodes
        self.classes = 2  # Number of classes
        low = 0.01
        high = 0.99

        # define base graph shapes
        # cross graph (shaped like X)
        self.crossAdj = np.array([[low,high,low,high,low,high,low,high,low],
                                  [low,low,high,low,low,low,low,low,low],
                                  [low,low,low,low,low,low,low,low,low],
                                  [low,low,low,low,high,low,low,low,low],
                                  [low,low,low,low,low,low,low,low,low],
                                  [low,low,low,low,low,low,high,low,low],
                                  [low,low,low,low,low,low,low,low,low],
                                  [low,low,low,low,low,low,low,low,high],
                                  [low,low,low,low,low,low,low,low,low]])
        # snowflake graph (shaped like *)
        self.snowflakeAdj = np.array([[low,high,high,high,high,high,high,high,high],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low],
                                      [low,low,low,low,low,low,low,low,low]])
        # circle graph (shaped like O, currently unused)
        self.circleAdj = np.array([[low,high,low,low,low,low,low,low,high],
                                   [low,low,high,low,low,low,low,low,low],
                                   [low,low,low,high,low,low,low,low,low],
                                   [low,low,low,low,high,low,low,low,low],
                                   [low,low,low,low,low,high,low,low,low],
                                   [low,low,low,low,low,low,high,low,low],
                                   [low,low,low,low,low,low,low,high,low],
                                   [low,low,low,low,low,low,low,low,high],
                                   [low,low,low,low,low,low,low,low,low]])
        # define labels
        self.cross_label = torch.tensor([0])
        self.snowflake_label = torch.tensor([1])
        self.circle_label = torch.tensor([2])

        # define identity matrix
        self.identity = torch.tensor(np.identity(self.num_nodes))

        # define permutation matrices
        # cyclicly permutes the nodes 0 > 1, 1 > 2, 2 > 3, ...
        self.cyclic = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 0]])

    def permute(self, adj, P):
        """
        Permute given matrix
        :param adj: adjancency matrix
        :param P: permutation matrix
        :return: permutated graph
        """
        return np.transpose(P) @ adj @ P

    def generate_graphs(self, base_graph, label, num_samples, mutate):
        """
        Generate a graph
        :param base_graph: topology of the graph
        :param label: label of the graph
        :param num_samples: number of graphs to generate
        :mutate: slightly randomize edges
        :return: None
        """
        if mutate:
            adj = np.copy(base_graph)
        else:
            adj = np.copy(int(base_graph))  # adjacency matrix

        for i in range(num_samples):  # generate samples and append to dataset
            edge_index, _ = dense_to_sparse(torch.bernoulli(torch.tensor(adj)))  # adjacency list
            graph = data.Data(x=torch.rand(self.num_nodes, self.num_nodes),
                              edge_index=edge_index,
                              edge_attr=self.identity,
                              y=label)
            self.synthetic_dataset.append(graph)

    def generate(self, num_samples=5000, mutate=True, permute_node_idx=True):
        """
        Main function (generate synthetic dataset)
        :param num_samples: number of samples
        :param mutate: slightly randomize edges
        :param permute_node_idx: permute node indices
        :return: datasets of graphs
        """
        self.synthetic_dataset = []

        if permute_node_idx:
            cross_adj = np.copy(self.crossAdj)

            for i in range(self.num_nodes):
                if i != 0:
                    cross_adj = self.permute(cross_adj, self.cyclic)
                self.generate_graphs(cross_adj, self.cross_label, int(num_samples / (2 * self.num_nodes)), mutate)

            snowflake_adj = np.copy(self.circleAdj)
            for i in range(self.num_nodes):
                if i != 0:
                    snowflake_adj = self.permute(snowflake_adj, self.cyclic)
                self.generate_graphs(snowflake_adj, self.snowflake_label, int(num_samples / (2 * self.num_nodes)), mutate)
        else:
            self.generate_graphs(self.crossAdj, self.cross_label, int(num_samples / 2), mutate)
            self.generate_graphs(self.circleAdj, self.snowflake_label, int(num_samples / 2), mutate)

        return WrapperSynthetic(self.synthetic_dataset, self.num_nodes, self.classes)


class WrapperSynthetic:

    def __init__(self, datasets, number_nodes_features, number_classes):
        """
        Wrapper for Synthetic dataset
        :param datasets: list dataset
        :param number_nodes_features: number of features per node
        :param number_classes: number of classes
        """
        self.dataset = datasets
        self.num_node_features = number_nodes_features
        self.num_classes = number_classes

    def __getitem__(self, item):
        """
        Get graph from dataset
        :param item: index item
        :return: graph
        """
        return self.dataset[item]

    def __len__(self):
        """
        Return size of the dataset
        :return: size of the dataset
        """
        return len(self.dataset)

    def shuffle(self):
        """
        Shuffle list
        :return: the object
        """
        random.shuffle(self.dataset)
        return self
