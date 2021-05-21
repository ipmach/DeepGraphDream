from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from abc import ABC
import torch


class DataClass(ABC):

    def __init__(self, dataset):
        """
        Datasets class
        :param dataset: dataset used from graph deep learning
        """
        self.dataset = dataset
        self.num_node_features = self.dataset.num_node_features
        self.num_classes = self.dataset.num_classes

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

    def shuffle_data(self, seed=12345):
        """
        Shuffle dataset
        :param seed: random seed
        :return: None
        """
        torch.manual_seed(seed)
        self.dataset = self.dataset.shuffle()

    def get_loader(self, train_percentage=0.75, batch_size=64):
        """
        Pass data to the loaders
        :param train_percentage: percentage to use in train
        :param batch_size: size of the batch size
        :return: train_loader, test_loader
        """
        train_dataset = self.dataset[:int(len(self.dataset) * train_percentage)]
        test_dataset = self.dataset[int(len(self.dataset) * train_percentage):]

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def view_graph_index(self, index, with_labels=True, figsize=(6,6)):
        """
        Visualize graph from index
        :param index: index of geometric data graph
        :param with_labels: show node labels
        :param figsize: (weight, height) of the plot
        :return: None
        """
        DataClass.view_graph(self[index], with_labels=with_labels, figsize=figsize)

    @staticmethod
    def view_graph(data, with_labels=True, figsize=(6,6)):
        """
        Visualize graph
        :param data: geometric data graph
        :param with_labels: show node labels
        :param figsize: (weight, height) of the plot
        :return: None
        """
        # obtain data
        edge = data.edge_index.numpy()  # Edge indeces
        X = data.x.numpy()  # Feature values
        y = data.y.numpy()[0]  # Label data
        # create nxgraph instance
        g = nx.Graph()
        g.add_edges_from(edge.T)
        # save labels
        labels = {}
        for j,x in enumerate(X):
          labels[j] = x
        # print graph information
        edges = str(g.number_of_edges())
        edges_space = "".join([" " for _ in range(4 - len(edges))])
        nodes = str(g.number_of_nodes())
        nodes_space = "".join([" " for _ in range(5 - len(nodes))])
        print("###########################")
        print("# Graph visualization     #")
        print("#                         #")
        print("#   Number of edges: " + edges + edges_space + "#")
        print("#   Number of nodes: " + nodes + nodes_space + "#")
        print("#   Label:", y, "             #")
        print("###########################")
        # plot graph
        plt.figure(figsize=figsize)
        nx.draw(g, labels=labels, with_labels=with_labels)
        return int(edges), int(nodes)