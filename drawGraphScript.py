import networkx as nx
from nxpd import draw
import matplotlib.pyplot as plt

from graphManipulation import read_graph_from_file, draw_graph


if __name__ == '__main__':
    G = read_graph_from_file('data/att48_d.txt', ' ')  # 'data/att48_xy.txt')

    draw_graph(G, [0, 1, 2])
    plt.show()
