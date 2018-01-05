import glob
import numpy as np
import networkx as nx
import numpy as np
import pytest

from aco import Ant_Colony
from graphManipulation import read_graph_from_file

@pytest.fixture
def simple_cube():
    G = nx.Graph()
    
    edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}), (4, 1, {'weight': 1}),
             (1, 3, {'weight': 1.41421}), (2, 4, {'weight': 1.41421})]
    G.add_edges_from(edges)

    return G
    
@pytest.fixture
def all_test_tsp():
    # get all files in data
    cases = glob.glob('data/*.txt')
    G = [None] * cases.__len__()
    for i, case_path in enumerate(cases):
        G[i] = read_graph_from_file(path=case_path, delimiter=' ')
    
    return G


def test_total_run_basic(simple_cube):
    """Start at 1 to make life easier"""
    colony = Ant_Colony(simple_cube, 50, 100, 1, 0.5, 0.2, True, 'TSP', 1)
    path_best, dist = colony.find()

    path_best = np.array(path_best)
    assert ((path_best == [1,2,3,4,1]).all() or (path_best == [1,4,3,2,1]).all()) and dist == 4,\
        'Best path not found.'
    
        
class Test_Consistency:
    
    @pytest.fixture
    def init_data(self):
        self.G = all_test_tsp()
        self.run_arbitrary = [None] * self.G.__len__()
        for i, graph in enumerate(self.G):
            colony = Ant_Colony(graph, 50, 1000, 1, 0.5, 0.2, True, 'TSP')
            self.run_arbitrary[i], _ = colony.find()
        
    def test_uniqueness(self, init_data):
        for i, path in enumerate(self.run_arbitrary):
            if path.__len__() > len(set(path)) + 1: # first node appear twice
                assert False, 'Path visited nodes more than once.'
        
        
    
        
