import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

import evaluation
from antcolony import AntColony
from graphManipulation import read_graph_from_file, draw_graph


def simple_cube():
    G = nx.Graph()
    edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}), (4, 1, {'weight': 1}),
             (1, 3, {'weight': 1.41421}), (2, 4, {'weight': 1.41421})]
    G.add_edges_from(edges)

    return G


def romanian_graph():  # this graph is not possible with unique visitable nodes
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

    if sys.argv.__len__() >= 13:
        """sys.argv:
        1: graph
        2: memory_filename
        3: algorithm
        4: goal
        5: ants_total
        6: iter
        7: beta
        8: rho
        9: init_pher
        10: q0
        11: rho_local
        12: unique_visit
        """
        G = read_graph_from_file('data/' + sys.argv[1], ' ')
        memory_filename = 'data/history/' + sys.argv[2] + '.npy'

        colony = AntColony(graph=G,
            ants_total=int(sys.argv[5]),
            iter=int(sys.argv[6]),
            alpha=1,
            beta=float(sys.argv[7]),
            rho=float(sys.argv[8]),
            init_pher=float(sys.argv[9]),
            q0=float(sys.argv[10]),
            unique_visit=bool(sys.argv[12]),
            goal=sys.argv[4],
            algo=sys.argv[3],
            rho_local=float(sys.argv[11]))

        shortest_path, shortest_dist, memory = colony.find(path=memory_filename)
    else:
        # G = romanian_graph()
        G = read_graph_from_file('data/us48.txt', ' ', file_xy_mat='data/coordinates/us48_xy.txt')
        #G = simple_cube()

        colony = AntColony(graph=G,
            ants_total=10,
            iter=2500,
            alpha=1,
            beta=2,
            rho=0.1,
            init_pher=(1/(48 * 12000)),
            q0=0.9,
            unique_visit=True,
            goal='TSP',
            algo='ACS',
            rho_local=0.1)

        add_info = 'paper_params'
        memory_filename = 'data/mem_' + add_info + '_algo-' + colony.algo + '_iter-' + str(colony.iter) + '_ants-' + str(colony.ants_total) + '.npy'
        shortest_path, shortest_dist, memory = colony.find(path=memory_filename)

        # evaluation.plot_distances(memory, title='Test', path='plots/test.pdf', show=False)
        # draw_graph(G, shortest_path, file_name='plots/graph.pdf')
        # plt.show()
