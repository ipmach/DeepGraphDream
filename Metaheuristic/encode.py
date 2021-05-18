import numpy as np


class NonEncode:

    def __init__(self):
        """
        Use as encode for some of the metaheuristics
        """
        self.dec2binv = lambda x: [self.dec2bin(i) for i in x]
        self.bin2decv = lambda x: [self.bin2dec(i) for i in x]

    def dec2bin(self, x):
        """
        Convert from space problem to metaheuristic encoding
        :param x: point in the problem space
        :return: point in the metaheuristic encoding
        """
        x = np.array(x)
        return "".join(list(x.astype(str)))

    def bin2dec(self, x):
        """
        Convert from metaheuristic encoding to space problem
        :param x: point in the metaheuristic encoding
        :return: point in the problem space
        """
        return np.array(list(x)).astype(int)