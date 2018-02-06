import networkx as nx


def simple_cube():
    G = nx.Graph()
    edges = [(1, 2, {'weight': 1}), (2, 3, {'weight': 1}), (3, 4, {'weight': 1}),
             (4, 1, {'weight': 1}), (1, 3, {'weight': 1.41421}),
             (2, 4, {'weight': 1.41421})]
    G.add_edges_from(edges)

    return G


def romanian_graph():
    '''This graph is not possible with unique visitable nodes, since the graph is not fully
    connected and has "dead ends"
    '''
    G = nx.Graph()
    edges = [(1, 2, {'weight': 71}), (2, 3, {'weight': 75}), (3, 4, {'weight': 118}),
             (4, 5, {'weight': 111}), (5, 6, {'weight': 70}), (6, 7, {'weight': 75}),
             (7, 8, {'weight': 120}), (8, 9, {'weight': 138}), (9, 10, {'weight': 101}),
             (10, 11, {'weight': 90}), (10, 12, {'weight': 85}), (12, 13, {'weight': 98}),
             (13, 14, {'weight': 86}), (12, 15, {'weight': 142}), (15, 16, {'weight': 92}),
             (16, 17, {'weight': 87}), (10, 18, {'weight': 211}), (18, 19, {'weight': 99}),
             (19, 1, {'weight': 151}), (3, 19, {'weight': 140}), (19, 20, {'weight': 80}),
             (18, 10, {'weight': 211}), (20, 8, {'weight': 146}), (20, 9, {'weight': 97})]

    G.add_edges_from(edges)
    return G
