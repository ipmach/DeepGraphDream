from torch_geometric.datasets import TUDataset
from Datasets.wrapper import WrapperSynthetic
from Datasets.data_abstract import DataClass
from tqdm.notebook import tqdm
import torch
import copy


class TUDData(DataClass):

    def __init__(self, name_dataset, reduce=False):
        """
        Datasets class for TUD datasets
        :param name_dataset: name dataset to load from TUDataset
        :param reduce: remove redundant edges and make then uppertriangle
        """
        dataset = TUDataset(root='data/TUDataset', name=name_dataset)
        if reduce:
            new_dataset = []
            for i in tqdm(range(len(dataset))):
                aux_graph = copy.deepcopy(dataset[i])
                aux_graph.edge_index = TUDData.reduce_edges(aux_graph.edge_index)
                new_dataset.append(copy.deepcopy(aux_graph))
            dataset = WrapperSynthetic(new_dataset, dataset.num_node_features,
                                       dataset.num_classes, None)
        super(TUDData, self).__init__(dataset)

    @staticmethod
    def reduce_edges(edge_index):
        """
        Remove redundant edges and make then uppertriangle
        :param edge_index: index edge
        """
        new_edge_index = {}
        for edge in edge_index.T:
            if edge[0] > edge[1]:
                edge = torch.flip(edge, [-1])
            edge = edge.detach().numpy()
            name = str(edge[0]) + str(edge[1])
            if name  not in new_edge_index.keys():
                new_edge_index[name] = edge

        return torch.tensor(list(new_edge_index.values())).T
