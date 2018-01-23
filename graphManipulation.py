import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def read_graph_from_file(path, delimiter, path_xy=None):
    """ Reads graph from file. Unrecognised lines should be commented by a #."""
    G = nx.Graph()
    dist_matrix = np.loadtxt(path, dtype='f', delimiter=delimiter)

    n = dist_matrix.shape[0]
    for i in range(n):
        for j in range(n)[i:]:
            G.add_edge(i, j, weight=dist_matrix[i,j])
                
    if path_xy is not None:
        xy_mat = np.loadtxt(path_xy, dtype='f', delimiter=delimiter)
        nx.set_node_attributes(G, (None, None), 'pos')
        for i in range(n):
            G.node[i]['pos'] = (xy_mat[i,0], xy_mat[i,1])
            
    return G


def add_visited_attribute_to_edge(G, path):
    nx.set_edge_attributes(G, False, 'visited')
    for i in range(path.__len__() - 1):
        G[path[i]][path[i + 1]]['visited'] = True

    return G


def draw_graph(G, file_name='graph.pdf'):
    """Simple representation of the graph featuring labels of a given attribute."""
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G,'pos')
    visited = nx.get_edge_attributes(G, 'visited')

    for key, value in visited.items():
        if value:
            visited[key] = 1
        else:
            visited[key] = 0

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.4,
                           edge_color=list(visited.values()),
                           edge_vmin=0,
                           edge_vmax=1)

    nx.draw_networkx_nodes(G, pos, node_size=50)

    plt.axis('off')
    plt.savefig(file_name)