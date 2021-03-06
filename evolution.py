import numpy as np
from copy import deepcopy
from antcolony import AntColony
from run_aco import simple_cube, read_graph_from_file
from heapq import *
import evaluation
import os


class Evolution():
    '''
        class to let several ant colonies compete against each other to determine good parameters for the evaluated
        algorithm
    '''

    def __init__(self, colonies, ants, algo, iter, init_params, graph, unique_visit, goal,
                 start_node, end_node, tries, epochs, variation, drop_out):

        '''
            initialize evolution.

            @param num_colonies (int): number of ant colonies
            @param num_ants (int): number of ants per colony
            @param algo (str): algorithm all (!) colonies follow
            @param iter (int): number of iterations each colony does to find a path
            @param init_genes (dict): initial parameters for the algorithm. They need to match the algorithm!
            @param graph (networkX graph): graph that all colonies explore
            @param unique_visit (bool): whether an ant can visit a single node more than once
            @param goal (str): problem to solve. {TSP, min_path}
            @param start (int): starting node
            @param end (int): destination node
            @param tries (int): ties each colony gets to solve the goal problem
            @param epochs (int): number of selection/variation iterations
            @param variation (float): (0 < x <1) fraction in whose boundaries a parameter may mutate upon a single variation
            @param drop_out (float): (0 < x < 1) fraction of all colonies that dies in each epoch
            @param count (int): running count acting as a colony id
        '''

        # ants
        self.num_colonies = colonies
        self.num_ants = ants
        self.algo = algo
        self.iter = iter
        self.init_genes = init_params

        # world
        self.graph = graph
        self.unique_visit = unique_visit
        self.goal = goal
        self.start = start_node
        self.end = end_node

        # evolution
        self.tries = tries
        self.epochs = epochs
        self.variation = variation
        self.drop_out = drop_out
        self.count = 0

        # the 6th day
        print('genesis')
        self.population = []
        for _ in range(self.num_colonies):
            colony, genes, count = self.make_colony(self.init_genes)
            fitness = self.eval_fitness(colony)
            heappush(self.population, (fitness, count, colony, genes))

    def make_colony(self, parent_genes):
        '''
            function to initialize a colony from a set of parent genes.

            @param parent_genes (dict): dictionary like init_params

            @return antcolony object
            @return genes (dict)
            @return id (current count)
        '''

        # variation
        genes = deepcopy(parent_genes)
        for key in parent_genes.keys():
            if parent_genes[key] is not None:
                factor = np.random.uniform(1 - self.variation, 1 + self.variation)
                genes[key] *= factor

        # initialization
        child = AntColony(graph=self.graph, ants_total=self.num_ants, iter=self.iter,
                          alpha=genes['alpha'], beta=genes['beta'], rho=genes['rho'],
                          unique_visit=self.unique_visit, goal=self.goal, start_node=self.start,
                          end_node=self.end, init_pher=genes['init_pher'],
                          min_pher=genes['min_pher'], max_pher=genes['max_pher'], q0=genes['q0'],
                          tau=genes['tau'], algo=self.algo)
        count = self.count
        # print('colony', self.count, 'is born')
        self.count += 1

        return child, genes, count

    def eval_fitness(self, colony):
        '''
            function to evaluate a colony's fitness

            @param colony (AntColony instance)

            @return average performance of all tries
        '''

        solutions = []
        for i in range(self.tries):
            _, dist, _ = colony.find
            solutions.append(dist)

        return np.mean(np.array(solutions))

    def begin(self, path=None):
        '''
            function to run the evolutionary process.

            @param path: path to safe memory

            @return alpha_colony (AntColony instance): colony that performed best overall
            @return alpha_genes (dict): parameters of this colony
            @return memory (ndarray): of shape (epochs, [average best distance, its std, epochs best distance, best colony's genes (dict)])
        '''

        mean_distances = []
        shortest_distances = []

        if path and os.path.exists(path):
            raise IOError('path already exists. choses another to save history.')

        memory = np.full((self.epochs, 4), None)

        for i in range(self.epochs):

            print('\nEpoch:', i)

            # selection
            num_survivors = int(self.num_colonies * (1 - self.drop_out))
            parents = nsmallest(num_survivors, self.population)

            # reproduction and evaluation
            children = []

            # make sure enough offspring is produced in case drop_out > 0.5
            while len(children) < self.num_colonies - len(parents):

                # most successful colony reproduces most often
                for parent in parents:

                    # keep total population
                    if len(children) == self.num_colonies - len(parents):
                        break

                    print('colony', parent[1], 'is reproducing')

                    # birth
                    parent_genes = parent[3]
                    child, child_genes, child_count = self.make_colony(parent_genes)
                    fitness = self.eval_fitness(child)
                    heappush(children, (fitness, child_count, child, child_genes))

            self.population = parents + children
            heapify(self.population)

            # epochs shortest distance
            epoch_winner = self.population[0]
            shortest_distances.append(epoch_winner[0])

            # epochs mean distance and std
            all_distances = np.array([self.population[i][0] for i in range(self.num_colonies)])
            mean = np.mean(all_distances)
            std = np.std(all_distances)
            mean_distances.append([mean, std])

            print([(self.population[i][0], self.population[i][1]) for i in range(len(self.population))])

            # saving history
            memory[i, 0] = mean
            memory[i, 1] = std
            memory[i, 2] = epoch_winner[0]
            memory[i, 3] = epoch_winner[3]
            if path:
                np.save(path, memory)

        # leaves the heap invariant
        alpha_distance, alpha_id, alpha_colony, alpha_genes = heappop(self.population)

        return alpha_colony, alpha_genes, memory


if __name__ == '__main__':

    gene_root = {'alpha': 1, 'beta': 1, 'rho': 0.1, 'init_pher': None, 'min_pher': None,
                 'max_pher': None, 'q0': None, 'tau': None}

    # G = simple_cube()
    G = read_graph_from_file(path='data/oliver30.txt', delimiter=' ')

    evolution = Evolution(colonies=2, ants=10, algo='ant_system', iter=10, init_params=gene_root,
                          graph=G, unique_visit=True, goal='TSP', start_node=None, end_node=None,
                          tries=1, epochs=10, variation=0.5, drop_out=0.5)

    alpha_colony, alpha_genes, memory = evolution.begin(path='data/evo_hist.npy')
    # evaluation.plot_evolution_hist(shortest_distances, mean_distances, path='plots/evo_test.pdf',
    #                                title='Evo Test')

    # print('\nwinner:')
    # print(alpha_colony.shortest_dist)
    # print(alpha_genes)
