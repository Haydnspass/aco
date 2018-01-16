import numpy as np
import pylab as plt
import os

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

    plt.plot(x, y)
    plt.xlabel('Iteration')
    plt.ylabel('Shortest Distance')

    #TODO: forcing x-labels to be integer

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

def plot_evolution_hist(shortest_distances, mean_distances, show=True, path=None, title=None):
    '''function to plot the history of an evolutionary process

     Args:
        shortest_distances (list): shortest distance (average over number of tries) found in each epoch
        mean_distance (2d list): mean distance and standard deviation of each epoch
        show (bool): whether to show the plot
        path (string): path to save the figure to
        title (string): plot title
    '''

    x = np.linspace(0, len(shortest_distances)-1, len(shortest_distances))
    y1 = np.array(shortest_distances)
    y2 = np.array(mean_distances)[:, 0]
    err2 = np.array(mean_distances)[:, 1]

    plt.plot(x, y1, '--o', label='shortest distance')
    plt.errorbar(x, y2, yerr=err2, fmt='--x', label='mean distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.legend()

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