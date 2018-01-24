import numpy as np
import pylab as plt
import os

def plot_distances(memories, labels, best_d=None, n_iter=None , show=True, path=None, title=None):
    '''plots the shortest distances of each iteration for all iterations

    Args:
        memories (list): of ndarrays as returned by colony.find
        show (bool): whether to show the plot
        path (string): path to save the figure to
        title (string): title for the plot
    returns:
        nothing
    '''

    distances = []
    for i in range(len(memories)):
        distances.append(memories[i][:, 0][np.where(memories[i][:, 0] != None)])

    if not n_iter:
        n_iter = min([len(distances[i]) for i in range(len(distances))])

    for i in range(len(distances)):
        distances[i] = distances[i][0:n_iter]
        if best_d:
            distances[i] /= best_d

    x = np.linspace(0, n_iter - 1, n_iter)

    for i in range(len(distances)):
        plt.plot(x, distances[i], label=labels[i])

    plt.xlabel('Iteration')
    if n_iter < 25:
        plt.xticks(range(0, n_iter))

    if best_d:
        plt.ylabel('Shortest Distance / Optimal Distance')
    else:
        plt.ylabel('Shortest Distance')

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

    #m = np.load('data/evo_Elitist_hist.npy')
    #plot_evolution_hist(m, path='plots/elitist_evo.pdf', title='evolutionary development of Elitist')

    m1 = np.load('data/us_ACS{a=1,b=5,r=0.05,t=1e-4,q0=0.1}.npy')
    m2 = np.load('data/us_elitist{a=1,b=5,r=0.3,t0=0.1.npy')
    m3 = np.load('data/us_AS{a=1,b=5,r=0.05.npy')

    plot_distances([m1, m2, m3], labels=['ACS', 'Elitist', 'AS'], best_d=36000, n_iter=10)

    #path='plots/US48_comparison_no_title.pdf'