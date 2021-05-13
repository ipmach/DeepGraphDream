from Datasets.SyntheticGenerator import SyntheticGenerator
from Datasets.data_abstract import DataClass


class SyntheticData(DataClass):

    def __init__(self, num_samples, permute=True):
        """
        Datasets class for Synthetic datasets
        :param num_samples: size dataset
        :param permute: do permutation
        """
        self.generator = SyntheticGenerator()
        dataset = self.generator.generate(num_samples=num_samples, permute=permute)
        super(SyntheticData, self).__init__(dataset)
