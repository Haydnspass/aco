import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import evaluation
from multiprocessing import Pool

from antcolony import AntColony
from graphManipulation import read_graph_from_file, draw_graph


def run_alg(i):
    # G = romanian_graph()
    G = read_graph_from_file('data/us48.txt', ' ', file_xy_mat='data/coordinates/us48_xy.txt')
    #G = simple_cube()

    colony = AntColony(graph=G, 
        ants_total=i,
        iter=12000 // i,
        alpha=1, 
        beta=2,
        rho=0.1,
        init_pher=(1/(48 * 12000)),
        q0=0.9,
        unique_visit=True, 
        goal='TSP', 
        algo='ACS')
    # colony = AntColony(G, 30, 2, 5, 1, 0.2, True, 'TSP', min_pher=0.001, max_pher=10, algo='min_max')
    # colony = AntColony(G, 20, 1000, 3, 1, 0.4, True, 'PathMin', 4, 10)
    add_info = 'multi'
    memory_filename = 'data/mem_' + add_info + '_algo-' + colony.algo + '_iter-' + str(colony.iter) + '_ants-' + str(colony.ants_total) + '.npy'
    shortest_path, shortest_dist, memory = colony.find(path=memory_filename)

if __name__ == '__main__':
    p = Pool(4)
    print(p.map(run_alg, [12, 24, 48]))
