from torch_geometric.utils import dense_to_sparse
from torch_geometric import data
import numpy as np
import torch


class SyntheticGenerator:

    def __init__(self):

        self.num_nodes = 9  # all graphs have exactly 9 nodes

        # define base graph shapes
        # cross graph (shaped like X)
        self.crossAdj = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0],
                                  [1, 0, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 1, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0]])
        # snowflake graph (shaped like *)
        self.snowflakeAdj = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0, 0, 0, 0, 0]])
        # circle graph (shaped like O, currently unused)
        self.circleAdj = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1],
                                   [1, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 1, 0]])
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

    def generate_graphs(self, base_graph, label, num_samples):
        """
        Generate a graph
        :param base_graph: topology of the graph
        :param label: label of the graph
        :param num_samples: number of graphs to generate
        :return: None
        """
        adj = np.copy(base_graph)  # adjacency matrix
        edge_index, _ = dense_to_sparse(torch.tensor(adj))  # adjacency list
        # print(num_samples)
        for i in range(num_samples):  # generate samples and append to dataset
            graph = data.Data(x=torch.rand(self.num_nodes, self.num_nodes),
                              edge_index=edge_index,
                              edge_attr=self.identity,
                              y=label)
            self.synthetic_dataset.append(graph)

    def generate(self, num_samples=5000, permute=False):
        """
        Main function (generate synthetic dataset)
        :param num_samples: number of samples
        :param permute: do permutation
        :return: datasets of graphs
        """
        self.synthetic_dataset = []

        if permute:
            cross_adj = np.copy(self.crossAdj)

            for i in range(self.num_nodes):
                if i != 0:
                    cross_adj = self.permute(cross_adj, self.cyclic)
                self.generate_graphs(cross_adj, self.cross_label, int(num_samples / (2 * self.num_nodes)))

            snowflake_adj = np.copy(self.snowflakeAdj)
            for i in range(self.num_nodes):
                if i != 0:
                    snowflake_adj = self.permute(snowflake_adj, self.cyclic)
                self.generate_graphs(snowflake_adj, self.snowflake_label, int(num_samples / (2 * self.num_nodes)))
        else:
            self.generate_graphs(self.crossAdj, self.cross_label, int(num_samples / 2))
            self.generate_graphs(self.snowflakeAdj, self.snowflake_label, int(num_samples / 2))

        return self.synthetic_dataset
