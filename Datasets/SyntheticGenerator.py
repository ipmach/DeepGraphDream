from torch_geometric.utils import dense_to_sparse
from torch_geometric import data
import numpy as np
import random
import torch


class SyntheticGenerator:

    def __init__(self):

        self.num_nodes = 9  # all graphs have exactly 9 nodes
        self.classes = 2  # Number of classes
        self.edge_weight_list = []  # optional biased edge weights (list of lists)
        low = 0.01  # probability of generating false edge
        high = 0.99  # probability of generating true edge

        # define base graph shapes
        # circle graph (shaped like O)
        self.circleAdj = np.array([[0,high,low,low,low,low,low,low,high],
                                   [0,0,high,low,low,low,low,low,low],
                                   [0,0,0,high,low,low,low,low,low],
                                   [0,0,0,0,high,low,low,low,low],
                                   [0,0,0,0,0,high,low,low,low],
                                   [0,0,0,0,0,0,high,low,low],
                                   [0,0,0,0,0,0,0,high,low],
                                   [0,0,0,0,0,0,0,0,high],
                                   [0,0,0,0,0,0,0,0,0]])

        # cross graph (shaped like X)
        self.crossAdj = np.array([[0,high,low,high,low,high,low,high,low],
                                  [0,0,high,low,low,low,low,low,low],
                                  [0,0,0,low,low,low,low,low,low],
                                  [0,0,0,0,high,low,low,low,low],
                                  [0,0,0,0,0,low,low,low,low],
                                  [0,0,0,0,0,0,high,low,low],
                                  [0,0,0,0,0,0,0,low,low],
                                  [0,0,0,0,0,0,0,0,high],
                                  [0,0,0,0,0,0,0,0,0]])
        # snowflake graph (shaped like *, currently unused)
        self.snowflakeAdj = np.array([[0,high,high,high,high,high,high,high,high],
                                      [0,0,low,low,low,low,low,low,low],
                                      [0,0,0,low,low,low,low,low,low],
                                      [0,0,0,0,low,low,low,low,low],
                                      [0,0,0,0,0,low,low,low,low],
                                      [0,0,0,0,0,0,low,low,low],
                                      [0,0,0,0,0,0,0,low,low],
                                      [0,0,0,0,0,0,0,0,low],
                                      [0,0,0,0,0,0,0,0,0]])
        # define labels
        self.circle_label = torch.tensor([0])
        self.cross_label = torch.tensor([1])
        self.snowflake_label = torch.tensor([2])

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

    def generate_graphs(self, base_graph, label, num_samples, mutate, edge_weights):
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
            adj = np.copy(np.round(base_graph))  # adjacency matrix

        for i in range(num_samples):  # generate samples and append to dataset
            if not mutate:
              edge_index, _ = dense_to_sparse(torch.tensor(adj))
            else:
              mutated_sample = torch.bernoulli(torch.tensor(adj))
              edge_index, _ = dense_to_sparse(mutated_sample)  # adjacency list
            graph = data.Data(x=torch.rand(self.num_nodes, self.num_nodes),
                              edge_index=edge_index,
                              edge_attr=self.identity,
                              y=label)
            self.synthetic_dataset.append(graph)
            if edge_weights:
              weights = 0.1*np.ones(len(edge_index.T))
              intended_edges = torch.tensor(np.copy(np.round(base_graph)))
              intended_index, _ = dense_to_sparse(intended_edges)
              #print(intended_index)

              for i,edge in enumerate(edge_index.T):
                #print("edge",edge.tolist())
                edge_found = False
                for intended_edge in intended_index.T:
                  #print(intended_edge.tolist())
                  if edge.tolist() == intended_edge.tolist():
                    #print("true")
                    edge_found = True
                    break
                if edge_found:
                  weights[i] = 1.

              self.edge_weight_list.append(weights)




    def generate(self, num_samples=5000, mutate=True, permute_node_idx=True, edge_weights=False):
        """
        Main function (generate synthetic dataset)
        :param num_samples: number of samples
        :param mutate: slightly randomize edges
        :param permute_node_idx: permute node indices
        :param edge_weights: return edge weights
        :return: datasets of graphs
        """
        self.synthetic_dataset = []

        if permute_node_idx:

            circle_adj = np.copy(self.circleAdj)
            for i in range(self.num_nodes):
                if i != 0:
                    circle_adj = self.permute(circle_adj, self.cyclic)
                self.generate_graphs(circle_adj, self.circle_label, int(num_samples / (2 * self.num_nodes)), mutate, edge_weights)

            cross_adj = np.copy(self.crossAdj)
            for i in range(self.num_nodes):
                if i != 0:
                    cross_adj = self.permute(cross_adj, self.cyclic)
                self.generate_graphs(cross_adj, self.cross_label, int(num_samples / (2 * self.num_nodes)), mutate, edge_weights)

        else:
            self.generate_graphs(self.crossAdj, self.cross_label, int(num_samples / 2), mutate, edge_weights)
            self.generate_graphs(self.circleAdj, self.snowflake_label, int(num_samples / 2), mutate, edge_weights)

        return WrapperSynthetic(self.synthetic_dataset, self.num_nodes, self.classes, self.edge_weight_list)


class WrapperSynthetic:

    def __init__(self, datasets, number_nodes_features, number_classes, edge_weight_list):
        """
        Wrapper for Synthetic dataset
        :param datasets: list dataset
        :param number_nodes_features: number of features per node
        :param number_classes: number of classes
        :param edge_weight_list: edge weights list
        """
        self.dataset = datasets
        self.num_node_features = number_nodes_features
        self.num_classes = number_classes
        self.edge_weights = edge_weight_list

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
