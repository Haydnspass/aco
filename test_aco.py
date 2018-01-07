import glob
import numpy as np
import networkx as nx
import numpy as np
import pytest

from antcolony import AntColony
from graphManipulation import read_graph_from_file

'''Fixtures providing various tests sets.'''

@pytest.fixture
def list_of_alg():
    return ['ant_system', 'elitist', 'min_max', 'ACS']
    
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
    
@pytest.fixture
def all_test_tsp_run():
    G = all_test_tsp()
    no_of_alg = list_of_alg().__len__()
    paths = [None] * G.__len__() * no_of_alg
    dists = [None] * G.__len__() * no_of_alg
    i = 0
    for graph in G:
        for alg in list_of_alg():
            if alg == 'min_max':
                colony = AntColony(graph, 50, 1, 1, 0.5, 0.2, True, 'TSP', min_pher=0.0001, max_pher=10, algo=alg)
            else:
                colony = AntColony(graph, 50, 1, 1, 0.5, 0.2, True, 'TSP', algo=alg)
                
            paths[i], dists[i] = colony.find()
            i += 1
        
    return paths, dists
    
'''Test methods.'''
def closeness_of_path(path):
    assert path[0] == path[-1], 'Path is not closed.'
    

def uniqueness_of_path(path):
    if path.__len__() > len(set(path)) + 1: # first node appear twice
        assert False, 'Path visited nodes more than once.'


'''Actual run of tests.'''
def test_total_run_basic(simple_cube):
    """Start at 1 to make life easier"""
    colony = AntColony(simple_cube, 4, 5, 1, 1, 1, True, 'TSP', start_node=1, algo='ant_system')
    path_best, dist = colony.find()

    path_best = np.array(path_best)
    assert ((path_best == [1,2,3,4,1]).all() or (path_best == [1,4,3,2,1]).all()) and dist == 4,\
        'Best path not found.'


def test_all_tsp(all_test_tsp_run):
    paths = all_test_tsp_run[0]
    dists = all_test_tsp_run[1]
    for i, path in enumerate(paths):
        closeness_of_path(path)
        uniqueness_of_path(path)
