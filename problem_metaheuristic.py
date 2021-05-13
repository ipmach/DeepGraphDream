from MetaHeuristics.Problems.interface_problem import Problem_Interface
import numpy as np
import torch


class Problem(Problem_Interface):
    """
    Class adapter to use the metaheuristics.
    Dependency:
        https://github.com/ipmach/MetaHeuristics.git
    """

    def __init__(self, graph, batch, model, index, weights_size):
        """
        Class adapter
        :param graph: target graph
        :param batch: batch of the graph
        :param model: model used
        :param index: index of last layer neuron
        :param weights_size: numbers of the edges weights
        """
        self.graph = graph
        self.batch = batch
        self.model = model
        self.index = index
        self.weights_size = weights_size

    def give_random_point(self):
        """
        Return a random point in the space
        :return: random point
        """
        return np.random.choice([0, 1], size=self.weights_size)

    def generate_space(self):
        """
        Not used
        :return:
        """
        raise Exception("Not implement it")

    def generate_neighbour(self, x):
        """
        Generate a neighbour from our point x
        :param x: point in our problem space
        :return: new point
        """
        return self.perturbation(x)

    def all_neighbours(self, x, size_sample=10):
        """
        Return a sample of neighbours points
        :param x: point in our problem space
        :param size_sample: size sample neighbours
        :return: sample of new points
        """
        neighbours = []
        for i in range(size_sample):
            neighbours.append(self.generate_neighbour(x))
        return neighbours

    def perturbation(self, solution, p=2):
        """
        Apply a perturbation in our problem space
        :param solution: point in our problem space
        :param p: level of perturbation
        :return: new point after the perturbation
        """
        indices = np.random.choice(np.arange(self.weights_size), size=p,
                                   replace=False)
        new_solution = solution.copy()
        for index in indices:
            new_solution[index] = not solution[index]
        return new_solution

    def solve(self, x, real=False):
        """
        Apply model in our point in the space
        :param x: point in the problem space
        :param real: return real value or flip value (metaheuristics minimize not maximize)
        :return: solution
        """
        i = 1 if real else -1  # The original heuristic are minimizing
        return i * float(self.model(self.graph.x, self.graph.edge_index, self.batch,
                                    torch.Tensor(x))[0][self.index].detach().numpy())

    def __call__(self, x, multiple=False, real=False):
        """
        Apply model in a array or single point of the problem space
        :param x: point in the problem space
        :param multiple: return one or multiple
        :param real: return real value or flip value (metaheuristics minimize not maximize)
        :return:
        """
        if type(x[0]).__module__ == np.__name__:  # check if returns a list of arrays
            results = []
            for i in x:
                results.append(self.solve(i, real=real))
            return results
        else:
            return self.solve(x, real=real)


class NonEncode:

    def __init__(self):
        """
        Use as encode for some of the metaheuristics
        """
        self.dec2binv = lambda x: self.dec2bin(x)
        self.bin2decv = lambda x: self.bin2dec(x)

    def dec2bin(self, x):
        """
        Convert from space problem to metaheuristic encoding
        :param x: point in the problem space
        :return: point in the metaheuristic encoding
        """
        return "".join(list(x.astype(str)))

    def bin2dec(self, x):
        """
        Convert from metaheuristic encoding to space problem
        :param x: point in the metaheuristic encoding
        :return: point in the problem space
        """
        return np.array(list(x)).astype(int)