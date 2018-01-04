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

if __name__ == "__main__":

    G = read_graph_from_file(path='data/fri26.txt', delimiter=' ')
    #G = simple_cube()

    colony = Ant_Colony(G, 100, 10000, 0.2, 6, 0.4)
    shortest_path, shortest_dist = colony.find()
    print('Shortest path: ', shortest_path, ' dist: ', shortest_dist)
    # DrawGraph(G, 'r', 'pher')
    # plt.show()
