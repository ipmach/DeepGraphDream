from torch_geometric.datasets import TUDataset
from Datasets.data_abstract import DataClass


class TUDData(DataClass):

    def __init__(self, name_dataset):
        """
        Datasets class for TUD datasets
        :param name_dataset: name dataset to load from TUDataset
        """
        dataset = TUDataset(root='data/TUDataset', name=name_dataset)
        super(TUDData, self).__init__(dataset)
