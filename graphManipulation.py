import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS



def read_graph_from_file(file_dist_mat, delimiter, file_xy_mat=None):
    """ Reads graph from file. Unrecognised lines should be commented by a #."""
    G = nx.Graph()
    dist_matrix = np.loadtxt(file_dist_mat, dtype='f', delimiter=delimiter)

    n = dist_matrix.shape[0]
    for i in range(n):
        for j in range(n)[i:]:
            G.add_edge(i, j, weight=dist_matrix[i,j])

    if file_xy_mat is not None:
        xy_mat = np.loadtxt(file_xy_mat, dtype='f', delimiter=delimiter)
        nx.set_node_attributes(G, (None, None), 'pos')
    else:
        """If xy matrix is not provided, calculate coordinates from principal component analysis"""
        # xy_pca = PCA(n_components=2)
        # xy_mat = xy_pca.fit_transform(dist_matrix)
        xy_mds = MDS(n_components=2)
        xy_mat = xy_mds.fit_transform(dist_matrix)
        nx.set_node_attributes(G, (None, None), 'pos')
    for i in range(n):
        G.node[i]['pos'] = (xy_mat[i,0], xy_mat[i,1])
            
    return G


def draw_graph(G, path, file_name='plots/graph.pdf'):
    """Simple representation of the graph featuring labels of a given attribute."""
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G,'pos')

    edges_on_path = [None] * (path.__len__() - 1)
    for i in range(edges_on_path.__len__()):
        edges_on_path[i] = (path[i], path[i + 1])
    edges_on_path = tuple(edges_on_path)


    f = plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    nx.draw_networkx_edges(G, pos, alpha=0.8, edgelist=edges_on_path, edge_color='r')
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='k')

    plt.axis('off')
    plt.savefig(file_name)
