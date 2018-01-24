import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import evaluation

from antcolony import AntColony
from graphManipulation import read_graph_from_file,DrawGraph


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
    G = read_graph_from_file(path='data/us48.txt', delimiter=' ')
    #G = simple_cube()

    colony = AntColony(graph=G, ants_total=30, iter=20, alpha=1, beta=5, rho=0.05, tau=0.0001, q0=0.1,
                       unique_visit=True, goal='TSP', algo='ACS')

    # colony = AntColony(G, 30, 2, 5, 1, 0.2, True, 'TSP', min_pher=0.001, max_pher=10, algo='min_max')
    # colony = AntColony(G, 20, 1000, 3, 1, 0.4, True, 'PathMin', 4, 10)
    shortest_path, shortest_dist, memory = colony.find(path='data/us_ACS{a=1,b=5,r=0.05,t=1e-4,q0=0.1}.npy')
    evaluation.plot_distances([memory], labels=['elitist'])
    # print('Shortest path: ', shortest_path, ' dist: ', shortest_dist)
    # DrawGraph(G, 'r', 'pher')
    # plt.show()
