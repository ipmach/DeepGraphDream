from Datasets.SyntheticGenerator import SyntheticGenerator
from Datasets.data_abstract import DataClass


class SyntheticData(DataClass):

    def __init__(self, num_samples, mutate=True, permute_node_idx=True, edge_weights=False):
        """
        Datasets class for Synthetic datasets
        :param num_samples: size dataset
        :param mutate: slightly randomize edges
        :param permute_node_idx: permute node indices
        :param edge_weights: include edge weights
        """
        self.generator = SyntheticGenerator()
        dataset = self.generator.generate(num_samples=num_samples,
                                          mutate=mutate,
                                          permute_node_idx=permute_node_idx,
                                          edge_weights=edge_weights)
        super(SyntheticData, self).__init__(dataset)
