from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch


class Edges2binary:

    @staticmethod
    def find_indexes(graph, edge_index):
        """
        From graph to edge binary representation
        :param graph: graph object
        :param edge_index: all edge indexes
        :return: binary representation
        """
        aux = graph.edge_index.T
        indexes = []
        for j in range(len(aux)):
            indexes.append(np.nonzero([bool(torch.all(i)) for i in edge_index.T == aux[j]])[0][0])
        values = np.zeros(len(edge_index.T)).astype(int)
        values[indexes] = np.ones(len(indexes))
        return values

    @staticmethod
    def initialize_indexes(nodes):
        """
        Initialize all edge indexes
        :param nodes: number of nodes
        :return: all edge indexes
        """
        edge = []
        for i in range(nodes):
            for j in range(nodes):
                edge.append([i, j])
        return torch.tensor(edge).T

    @staticmethod
    def convert_edges(index, edge_index):
        """
        Convert binary representation to edges index
        :param index: binary representation
        :param edge_index: all edges indexes
        :return: edges index
        """
        aux = []
        for j, i in enumerate(index):
            if i:
                aux.append(edge_index.T[j].detach().numpy())
        return torch.tensor(aux).T


def score(old, new, index):
    """
    Score in the solution space
    :param old: original point in the solution space
    :param new: new point in the solution space
    :param index: index we want to optimize
    :return: score value
    """
    d = euclidean_distances(old.detach().numpy(), new.detach().numpy())[0][0]
    c = 1 if new[0][index] < old[0][index] else -1
    return c * (d / np.sqrt(2))