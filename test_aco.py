import numpy as np
import networkx as nx
import numpy as np
import pytest

from aco import Ant_Colony

@pytest.fixture
def simple_cube():
    G = nx.Graph()
    
    edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}), (4, 1, {'weight': 1}),
             (1, 3, {'weight': 1.41421}), (2, 4, {'weight': 1.41421})]
    G.add_edges_from(edges)

    return G


def test_total_run(simple_cube):
    """Start at 1 to make life easier"""
    colony = Ant_Colony(simple_cube, 50, 100, 1, 0.5, 0.2, True, 'TSP', 1)
    
    path_best, dist = colony.find()
    print(path_best)
    # order result such that 1 is in the beginning
    path_best = np.array(path_best)
    assert ((path_best == [1,2,3,4,1]).all() or (path_best == [1,4,3,2,1]).all()) and dist == 4,\
        'Best path not found.'
