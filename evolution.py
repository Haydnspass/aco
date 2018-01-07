import numpy as np
from copy import deepcopy
from antcolony import AntColony
from run_aco import simple_cube, read_graph_from_file
from heapq import *

class Evolution():
    ''' class to let several ant colonies compete agains each other to determine good parameters for the evaluated
    algorithm

        Attributes:
            num_colonies (int): number of ant colonies
            num_ants (int): number of ants per colony
            algo (str): algorithm all (!) colonies follow
            iter (int): number of iterations each colony does to find a path
            init_genes (dict): initial parameters for the algorithm. They need to match the algorithm!
            graph (networkX graph): graph that all colonies explore
            unique_visit (bool): whether an ant can visit a single node more than once
            goal (str): problem to solve. {TSP, min_path}
            start (int): starting node
            end (int): destination node
            tries (int): ties each colony gets to solve the goal problem
            epochs (int): number of selection/variation iterations
            variation (float): (0 < x <1) fraction in whose boundaries a parameter may mutate upon a single variation
            drop_out (float): (0 < x < 1) fraction of all colonies that dies in each epoch
            count (int): running count acting as a colony id
    '''

    def __init__(self, colonies, ants, algo, iter, init_params, graph, unique_visit, goal, start_node, end_node, \
                 tries, epochs, variation, drop_out):

        #ants
        self.num_colonies = colonies
        self.num_ants = ants
        self.algo = algo
        self.iter = iter
        self.init_genes = init_params

        #world
        self.graph = graph
        self.unique_visit = unique_visit
        self.goal = goal
        self.start = start_node
        self.end = end_node

        #evolution
        self.tries = tries
        self.epochs = epochs
        self.variation = variation
        self.drop_out = drop_out
        self.count = 0

        #the 6th day
        print('genesis')
        self.population = []
        for _ in range(self.num_colonies):
            colony, genes, count = self.make_colony(self.init_genes)
            fitness = self.eval_fitness(colony)
            heappush(self.population, (fitness, count, colony, genes))

    def make_colony(self, parent_genes):
        '''function to initialize a colony from a set of parent genes.

            Arguments:
                parent_genes (dict): dictionary like init_params
            Returns:
                antcolony object
                genes (dict)
                id (current count)
        '''

        #vaiation
        genes = deepcopy(parent_genes)
        for key in parent_genes.keys():
            if parent_genes[key] != None:
                factor = np.random.uniform(1 - self.variation, 1 + self.variation)
                genes[key] *= factor

        #initialization
        child = AntColony(graph=self.graph, ants_total=self.num_ants, iter=self.iter, \
                          alpha=genes['alpha'], beta=genes['beta'], rho=genes['rho'], unique_visit=self.unique_visit,\
                          goal=self.goal, start_node=self.start, end_node=self.end, init_pher=genes['init_pher'], \
                          min_pher=genes['min_pher'], max_pher=genes['max_pher'], q0=genes['q0'], tau=genes['tau'], \
                          algo=self.algo)
        count = self.count
        #print('colony', self.count, 'is born')
        self.count += 1

        return child, genes, count

    def eval_fitness(self, colony):
        '''function to evaluate a colony's fitness

        Arguments:
            colony (AntColony instance)
        Returns:
            average performance of all tries
        '''

        solutions = []
        for i in range(self.tries):
            _, dist = colony.find()
            solutions.append(dist)

        return np.mean(np.array(solutions))

    def begin(self):
        '''function to run the evolutionary process.

        Returns:
            alpha_colony (AntColony instance): colony that perfomed best overall
            alpha_genes (dict): parameters of this colony'''

        for i in range(self.epochs):

            print('\nEpoch:', i)

            #selection
            num_survivors = int(self.num_colonies * (1 - self.drop_out))
            parents = nsmallest(num_survivors, self.population)

            #reproduction and evaluation
            children = []

            #make sure enough offspring is produced in case drop_out > 0.5
            while len(children) < self.num_colonies - len(parents):

                #most successful colony reproduces most often
                for parent in parents:

                    #keep total population
                    if len(children) == self.num_colonies - len(parents):
                        break

                    print('colony', parent[1], 'is reproducing')

                    #birth
                    parent_genes = parent[3]
                    child, child_genes, child_count = self.make_colony(parent_genes)
                    fitness = self.eval_fitness(child)
                    heappush(children, (fitness, child_count, child, child_genes))

            self.population = parents + children
            heapify(self.population)

            print([(self.population[i][0], self.population[i][1]) for i in range(len(self.population))])

        alpha_distance, alpha_id, alpha_colony, alpha_genes = heappop(self.population)

        return alpha_colony, alpha_genes


if __name__ == '__main__':

    gene_root = {'alpha': 1, 'beta': 1, 'rho': 0.1, 'init_pher': None, 'min_pher': None, 'max_pher': None, 'q0': None, 'tau': None}

    #G = simple_cube()
    G = read_graph_from_file(path='data/oliver30.txt', delimiter=' ')

    evolution = Evolution(colonies=5, ants=15, algo='ant_system', iter=10, init_params=gene_root, graph=G, unique_visit=True, \
                    goal='TSP', start_node=None, end_node=None, tries=3, epochs=15, variation=0.1, drop_out=0.5)

    alpha_colony, alpha_genes = evolution.begin()
    print(alpha_colony.shortest_dist)
