import random


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