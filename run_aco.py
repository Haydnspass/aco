import networkx as nx
import numpy as np
from aco import Ant_Colony
import matplotlib.pyplot as plt

# https://github.com/MUSoC/Visualization-of-popular-algorithms-in-Python/blob/master/Travelling%20Salesman%20Problem/tsp_christofides.py
#takes input from the file and creates a weighted undirected graph


def CreateGraph():
    G = nx.Graph()
    dist_matrix = np.loadtxt("input.txt", dtype='f', delimiter=' ')
    n = dist_matrix.shape[0]
    for i in range(n):
        for j in range(n)[i:]:
            G.add_edge(i, j, weight=dist_matrix[i,j])
    return G


def DrawGraph(G, color):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True,
            edge_color=color)  # with_labels=true is to show the node number in the output graph
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)  # prints weight on all the edges
    return pos

if __name__ == "__main__":
    # G = nx.Graph()
    #
    # edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}), (4, 1, {'weight': 1}),
    #          (1, 3, {'weight': 1.41421}), (2, 4, {'weight': 1.41421})]
    # G.add_edges_from(edges)
    # nx.set_edge_attributes(G, 1, 'weight')
    G = CreateGraph()
    DrawGraph(G, 'r')
    # plt.show()

    nx.set_edge_attributes(G, 0, 'pher')
    colony = Ant_Colony(G, 50, 100, 0.5, 1)
    print(colony.find())
