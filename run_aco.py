import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from aco import Ant_Colony


"""takes input from the file and creates a weighted undirected graph.
Inspired by
https://github.com/MUSoC/Visualization-of-popular-algorithms-in-Python/blob/master/
Travelling%20Salesman%20Problem/tsp_christofides.py
"""


def read_graph_from_file(path, delimiter):
    """ Reads graph from file. Unrecognised lines should be commented by a #."""
    G = nx.Graph()
    dist_matrix = np.loadtxt(path, dtype='f', delimiter=delimiter)
    n = dist_matrix.shape[0]
    for i in range(n):
        for j in range(n)[i:]:
            G.add_edge(i, j, weight=dist_matrix[i,j])
            
    return G


def DrawGraph(G, color, attribute):
    """Simple representation of the graph featuring labels of a given attribute."""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, edge_color=color)
    edge_labels = nx.get_edge_attributes(G, attribute)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)  # prints weight on all the edges
    return pos


def simple_cube():
    G = nx.Graph()
    edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}), (4, 1, {'weight': 1}),
             (1, 3, {'weight': 1.41421}), (2, 4, {'weight': 1.41421})]
    G.add_edges_from(edges)

    return G


def romanian_graph(): # this graph is not possible with unique visitable nodes
    G = nx.Graph()
    edges = [(1, 2, {'weight': 71}), (2, 3, {'weight': 75}), (3, 4, {'weight': 118}), (4, 5, {'weight': 111}),
                 (5, 6, {'weight': 70}), (6, 7, {'weight': 75}), (7, 8, {'weight': 120}), (8, 9, {'weight': 138}),
                 (9, 10, {'weight': 101}), (10, 11, {'weight': 90}), (10, 12, {'weight': 85}), (12, 13, {'weight': 98}),
                 (13, 14, {'weight': 86}), (12, 15, {'weight': 142}), (15, 16, {'weight': 92}),
                 (16, 17, {'weight': 87}),
                 (10, 18, {'weight': 211}), (18, 19, {'weight': 99}), (19, 1, {'weight': 151}),
                 (3, 19, {'weight': 140}),
                 (19, 20, {'weight': 80}), (18, 10, {'weight': 211}), (20, 8, {'weight': 146}), (20, 9, {'weight': 97})]

    G.add_edges_from(edges)
    return G


if __name__ == "__main__":

    # G = romanian_graph()
    G = read_graph_from_file(path='data/oliver30.txt', delimiter=' ')
    # G = simple_cube()

    colony = Ant_Colony(G, 30, 100, 1, 0.2, 0.4, True, 'TSP')
    # colony = Ant_Colony(G, 20, 1000, 3, 1, 0.4, True, 'PathMin', 4, 10)
    shortest_path, shortest_dist = colony.find()
    print('Shortest path: ', shortest_path, ' dist: ', shortest_dist)
    # DrawGraph(G, 'r', 'pher')
    # plt.show()
