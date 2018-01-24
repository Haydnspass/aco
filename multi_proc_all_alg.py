import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import evaluation
from multiprocessing import Pool

from antcolony import AntColony
from graphManipulation import read_graph_from_file, draw_graph


def run_alg(args):
    (ants_total, iter, alpha, beta, rho, q0, algo) = args
    # G = romanian_graph()
    G = read_graph_from_file('data/us48.txt', ' ', file_xy_mat='data/coordinates/us48_xy.txt')
    #G = simple_cube()

    colony = AntColony(graph=G, 
        ants_total=ants_total,
        iter=iter,
        alpha=alpha, 
        beta=beta,
        rho=rho,
        init_pher=(1/(48 * 12000)),
        q0=q0,
        unique_visit=True, 
        goal='TSP', 
        algo=algo)
    # colony = AntColony(G, 30, 2, 5, 1, 0.2, True, 'TSP', min_pher=0.001, max_pher=10, algo='min_max')
    # colony = AntColony(G, 20, 1000, 3, 1, 0.4, True, 'PathMin', 4, 10)
    add_info = 'multi'
    memory_filename = 'data/mem_' + add_info + '_algo-' + colony.algo + '_ants-' \
        + str(colony.ants_total) + '_iter-' + str(colony.iter) \
        + '_alpha-' + str(colony.alpha)  + '_beta-' + str(colony.beta) \
        + '_rho-' + str(colony.rho)  + '_q0-' + str(colony.q0)+ '.npy'
    shortest_path, shortest_dist, memory = colony.find(path=memory_filename)

if __name__ == '__main__':
    p = Pool(40)
    print(p.map(run_alg, [(24, 2500, 1, 2, 0.1, 0.9, 'ACS'),
                            (24, 2500, 1, 2, 0.1, 0.9, 'ant_system'),
                            (24, 2500, 1, 2, 0.1, 0.9, 'elitist'),
                            (24, 2500, 1, 0, 0.1, 0.9, 'ACS'),
                            (24, 2500, 1, 0, 0.1, 0.9, 'ant_system'),
                            (24, 2500, 1, 0, 0.1, 0.9, 'elitist'),
                            (24, 2500, 1, 2, 0.4, 0.9, 'ACS'),
                            (24, 2500, 1, 2, 0.4, 0.9, 'ant_system'),
                            (24, 2500, 1, 2, 0.4, 0.9, 'elitist'),
                            (24, 2500, 1, 2, 0.9, 0.9, 'ACS'),
                            (24, 2500, 1, 2, 0.9, 0.9, 'ant_system'),
                            (24, 2500, 1, 2, 0.9, 0.9, 'elitist'),
                            (24, 2500, 1, 2, 0.01, 0.9, 'ACS'),
                            (24, 2500, 1, 2, 0.01, 0.9, 'ant_system'),
                            (24, 2500, 1, 2, 0.01, 0.9, 'elitist')
                            ]))
