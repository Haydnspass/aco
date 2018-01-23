import networkx as nx
from nxpd import draw
import matplotlib.pyplot as plt

from graphManipulation import read_graph_from_file, draw_graph, add_visited_attribute_to_edge


if __name__ == '__main__':
    G = read_graph_from_file(path='data/att48_d.txt', delimiter=' ', path_xy='data/att48_xy.txt')
    # position is stored as node attribute data for random_geometric_graph

    G = add_visited_attribute_to_edge(G, [0, 1, 2])

    draw_graph(G)
    plt.show()