import numpy as np
import pylab as plt
import os

def plot_distances(memory, show=True, path=None, title=None):
    '''plots the shortest distances of each iteration for all iterations

    Args:
        memory (ndarray): as returned by colony.find
        show (bool): whether to show the plot
        path (string): path to save the figure to
        title (string): title for the plot
    returns:
        nothing
    '''

    distances = memory[:, 0][np.where(memory[:, 0] != None)]
    x = np.linspace(0, len(distances) - 1, len(distances))

    plt.plot(x, distances)
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

def plot_evolution_hist(memory, show=True, path=None, title=None):
    '''function to plot the history of an evolutionary process

     Args:
        memory (ndarray): as returned by evolution.begin()
        show (bool): whether to show the plot
        path (string): path to save the figure to
        title (string): plot title
    '''

    n_epochs = len(memory[:, 0][np.where(memory[:, 0] != None)])
    x = np.linspace(0, n_epochs - 1, n_epochs)
    y1 = memory[0: n_epochs, 2]
    y2 = memory[0: n_epochs, 0]
    err2 = memory[0: n_epochs, 1]

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

if __name__ == '__main__':

    m = np.load('data/find_test.npy')
    plot_distances(m)