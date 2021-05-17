from MetaHeuristics.Problems.interface_problem import Problem_Interface
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch


class Problem(Problem_Interface):
    """
    Class adapter to use the metaheuristics.
    Dependency:
        https://github.com/ipmach/MetaHeuristics.git
    """

    def __init__(self, graph, batch, model, index, weights_size, solver):
        """
        Class adapter
        :param graph: target graph
        :param batch: batch of the graph
        :param model: model used
        :param index: index of last layer neuron
        :param weights_size: numbers of the edges weights
        :param solver: solver function use
        """
        self.graph = graph
        self.batch = batch
        self.model = model
        self.index = index
        self.weights_size = weights_size
        self.always_multiple = False
        self.solver = solver

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
        return i * self.solver(self.model, self.graph, self.batch, self.index, x)

    def __call__(self, x, multiple=False, real=False):
        """
        Apply model in a array or single point of the problem space
        :param x: point in the problem space
        :param multiple: return one or multiple
        :param real: return real value or flip value (metaheuristics minimize not maximize)
        :return:
        """
        if multiple or self.always_multiple:  # check if returns a list of arrays
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


def find_indeces(graph, edge_index):
  aux = graph.edge_index.T
  indeces = []
  for j in range(len(aux)):
      indeces.append(np.nonzero([bool(torch.all(i)) for i in edge_index.T == aux[j]])[0][0])
  values = np.zeros(len(edge_index.T)).astype(int)
  values[indeces] = np.ones(len(indeces))
  return values


def initialize_indeces(nodes):
    edge = []
    for i in range(nodes):
      for j in range(nodes):
        edge.append([i, j])
    return torch.tensor(edge).T


def conver_edges(index, edge_index):
    aux = []
    for j, i in enumerate(index):
      if i:
          aux.append(edge_index.T[j].detach().numpy())
    return torch.tensor(aux).T


def score(old, new, index):
    d = euclidean_distances(old.detach().numpy(), new.detach().numpy())[0][0]
    c = 1 if new[0][index] < old[0][index] else -1
    return c * (d / np.sqrt(2))


def solver(model, graph, batch, index, x):
    return float(model(graph.x, conver_edges(x), batch)[0][0])


