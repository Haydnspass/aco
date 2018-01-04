import numpy as np
import networkx as nx
import numpy as np
import pytest

from aco import Ant_Colony

def test_simple_cube():
    G = nx.Graph()
    
    edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}), (4, 1, {'weight': 1}),
             (1, 3, {'weight': 1.41421}), (2, 4, {'weight': 1.41421})]
    G.add_edges_from(edges)
    # order result such that 1 is in the beginning
    colony = Ant_Colony(G, 50, 100, 1, 0.5, 0.02, True, 'TSP')
    best_path, dist = colony.find()
    best_path = np.array(best_path)
    best_path = np.roll(best_path, shift=4-np.where(best_path == 1)[0][0])
    print('Shortest path: ', best_path, ' dist: ', dist)

    assert ((best_path == [1,2,3,4]).all() or (best_path == [1,4,3,2]).all()) and dist == 3
