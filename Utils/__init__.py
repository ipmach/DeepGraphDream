import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx


def view_graph_masked(data, mask, mask_weights=None, with_labels=True, figsize=(6,6)):
        """
        Visualize graph
        :param data: geometric data graph
        :param with_labels: show node labels
        :param figsize: (weight, height) of the plot
        :return: None
        """
        # obtain data
        edge = data.edge_index.numpy() # Edge indeces
        numpy_mask = mask.detach().numpy()
        edge = edge[:, numpy_mask]
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
        mask_weights = mask_weights[mask_weights>0]
        plt.figure(figsize=figsize)
        nx.draw(g, labels=labels, with_labels=with_labels, width=mask_weights)
        return int(edges), int(nodes)

def visualize_new_graph(initial_graph, mask):
    logic_mask = ((mask) * 1)
    discrete_mask = torch.nonzero(logic_mask).flatten()
    view_graph_masked(initial_graph, discrete_mask, mask.detach().numpy(), with_labels=False)
