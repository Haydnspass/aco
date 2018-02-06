import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

import evaluation
from antcolony import AntColony
from networkxgraphs import simple_cube, romanian_graph
from graphManipulation import read_graph_from_file, draw_graph

'''
    Jacqueline Wagner, Lucas Möller, Lucas-Raphael Müller

    Wrapper function to run ant colony optimization.
'''


if __name__ == "__main__":

    if sys.argv.__len__() >= 13:
        '''Run ant optimisation with parameters specified by the arguments in command line.
        sys.argv:
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
        '''

        G = read_graph_from_file('../data/' + sys.argv[1], ' ')
        memory_filename = '../data/history/' + sys.argv[2] + '.npy'

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
        '''If not enough arguments were passed, run this predefined example.
        '''
        G = read_graph_from_file('../data/us48.txt', ' ', file_xy_mat='../data/coordinates/us48_xy.txt')

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
        memory_filename = '../data/mem_' + add_info + '_algo-' + colony.algo + '_iter-' \
                          + str(colony.iter) + '_ants-' + str(colony.ants_total) + '.npy'
        shortest_path, shortest_dist, memory = colony.find(path=memory_filename)
