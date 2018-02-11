import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from multiprocessing import Pool

from antcolony import AntColony
import evaluation
from graphManipulation import read_graph_from_file, draw_graph

'''
    Jacqueline Wagner, Lucas Möller, Lucas-Raphael Müller

    Wrapper function to run ant colony optimization in parallel.
'''


def run_alg(args):
    (graph, ants_total, iter, alpha, beta, rho, q0, rho_local, del_min, del_max, tau0, algo, outFname) = args
    # G = romanian_graph()
    G = read_graph_from_file(delimiter=' ', file_xy_mat='../data/coordinates/' + graph + '.txt')

    colony = AntColony(graph=G,
                       ants_total=ants_total,
                       iter=iter,
                       alpha=alpha,
                       beta=beta,
                       rho=rho,
                       init_pher=tau0,
                       q0=q0,
                       min_pher=del_min,
                       max_pher=del_max,
                       unique_visit=True,
                       goal='TSP',
                       algo=algo,
                       rho_local=rho_local)
    # colony = AntColony(G, 30, 2, 5, 1, 0.2, True, 'TSP', min_pher=0.001, max_pher=10, algo='min_max')
    # colony = AntColony(G, 20, 1000, 3, 1, 0.4, True, 'PathMin', 4, 10)
    # add_info = 'multi' + '_graph-' + graph
    # memory_filename = '../data/mem_' + add_info + '_algo-' + colony.algo + '_ants-' \
    #     + str(colony.ants_total) + '_iter-' + str(colony.iter) \
    #     + '_alpha-' + str(colony.alpha) + '_beta-' + str(colony.beta) \
    #     + '_rho-' + str(colony.rho) + '_q0-' + str(colony.q0) + '_rho_loc-' \
    #     + str(colony.rho_local) + '.npy'
    shortest_path, shortest_dist, memory = colony.find(path=outFname + 'npy')


if __name__ == '__main__':
    p = Pool(30)
    # args = ('oliver30_xy', 30, 1000, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ant_system')
    # run_alg(args)
    # (graph, ants_total, iter, alpha, beta, rho, q0, rho_local, del_min, del_max, tau0, algo)
    print(p.map(run_alg, [('oliver30_xy', 30, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ant_system', 'f1'),
                          ('us48_xy', 48, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ant_system', 'f2'),
                          ('berlin52_xy', 52, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ant_system', 'f3'),
                          ('eil76_xy', 76, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ant_system', 'f4'),
                          ('gr202_xy', 202, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ant_system', 'f5'),
                          ('oliver30_xy', 30, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ACS', 'f6'),
                          ('us48_xy', 48, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ACS', 'f7'),
                          ('berlin52_xy', 52, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ACS', 'f8'),
                          ('eil76_xy', 76, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ACS', 'f9'),
                          ('gr202_xy', 202, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'ACS', 'f10'),
                          ('oliver30_xy', 30, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'elitist', 'f11'),
                          ('us48_xy', 48, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'elitist', 'f12'),
                          ('berlin52_xy', 52, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'elitist', 'f13'),
                          ('eil76_xy', 76, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'elitist', 'f14'),
                          ('gr202_xy', 202, 50, 1, 5, 0.4, 0.1, 0.4, 0, 0.006, 0.0001, 'elitist', 'f15')]))
