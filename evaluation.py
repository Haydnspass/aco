import numpy as np
import pylab as plt
import os
from matplotlib.ticker import MaxNLocator
def plot_distances(distances, show=True, path=None, title=None):
    '''plots the shortest distances of each iteration for all iterations

    Args:
        distances (list): list of shortest distance in each iteration, starting from index zero
        show (bool): whether to show the plot
        path (string): path to save the figure to
        title (string): title for the plot
    returns:
        nothing
    '''

    x = np.linspace(0, len(distances), len(distances))
    y = np.array(distances)
    print('number of distances', len(y))
    print('number of iterations', len(x))
    plt.plot(x, y)
    plt.xlabel('Iteration')
    plt.ylabel('Shortest Distance')

    #TODO: forching x-labels to be integer

    if title:
        plt.title(title)
    if path:
        if os.path.exists(path):
            raise IOError('file already exists. figure is not saved.')
        else:
            plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()

    return

