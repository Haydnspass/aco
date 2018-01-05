import networkx as nx
import numpy as np


def read_graph_from_file(path, delimiter):
    """ Reads graph from file. Unrecognised lines should be commented by a #."""
    G = nx.Graph()
    dist_matrix = np.loadtxt(path, dtype='f', delimiter=delimiter)
    n = dist_matrix.shape[0]
    for i in range(n):
        for j in range(n)[i:]:
            G.add_edge(i, j, weight=dist_matrix[i,j])
            
    return G
    
def DrawGraph(G, color, attribute):
    """Simple representation of the graph featuring labels of a given attribute."""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, edge_color=color)
    edge_labels = nx.get_edge_attributes(G, attribute)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11)  # prints weight on all the edges
    return pos
