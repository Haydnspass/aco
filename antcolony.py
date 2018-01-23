import networkx as nx
import numpy as np
import threading
import os

from ant import Ant


""" 
    Jacqueline Wagner, Lucas Möller, Lucas-Raphael Müller 
"""

class AntColony:
    """
        A class representing an ant colony.
    """

    def __init__(self, graph, ants_total, iter, alpha, beta, rho, unique_visit, goal, start_node=None, end_node=None, \
                 init_pher=0.1, min_pher=None, max_pher=None, q0=0.3, tau=0.00001, algo='ant_system'):

        """
            Initialize an ant colony.

            @param graph: A graph representation of the world.
            @param ants_total: Total number of ants.
            @param iter: Number of iterations.
            @param alpha: Power of the pheromone factor.
            @param beta: Power of the attractiveness factor.
            @param rho: Determines the evaporation factor.
            @param unique_visit: Determines whether a node can only be visited once.
            @param goal: Determines the goal of the optimisation. Could be TSP, i.e. must visit all cities at least once.
            @param start_node: Specifies the initial node.
            @param end_node: Specifies the end_node for path minimisation
            @param init_pher: Initial pheromone value.
            @param min_pher: Minimal pheromon value for max-min ant optimisation.
            @param max_pher: Maximal pheromon value for max-min ant optimisation.
            @param q0: specifies ratio between exploitation and exploration.
            @param tau: initial pheromon value for ACS.
            @param algo: Speciefies ant algorithm (ant_system, elitist, min_max, ACS).
        """

        self.graph = graph
        self.ants_total = ants_total
        self.iter = iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.tau = tau
        self.init_pher = init_pher
        self.min_pher = min_pher
        self.max_pher = max_pher
        self.algo = algo
        self.unique_visit = unique_visit
        self.goal = goal
        self.start_node = start_node
        self.end_node = end_node

        self.best_ant = None
        self.shortest_dist = None
        self.shortest_path = None
        nx.set_edge_attributes(self.graph, 0, 'pher')  # initialize pheromone values

        self.ants = self.init_ants()

    def init_ants(self, ants=None):
        """
            Initialize array of ants.

            @called in: __init__(), find()

            @param ants:

            @return ants: The arary of ants.
        """
              
        ants = [None] * self.ants_total
        for i in range(self.ants_total):
            """
                For TSP ants are placed randomly, for shortest path a starting node needs to be specified beforehand.
            """
            if self.start_node is None and self.goal == 'TSP':
                self.start_node = np.random.choice(self.graph.nodes())
                
            ants[i] = Ant(colony=self, graph=self.graph, init_loc=self.start_node, alpha=self.alpha, beta=self.beta, \
                          unique_visit=self.unique_visit, goal=self.goal, end_node=self.end_node)
            
        return ants

    def init_pheromone(self):
        """
            Initializes pheromone amounts for each edge of the graph.

            @called in: update_pheromone()
        """
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['pher'] = self.init_pher

    def find(self, path=None):
        """Start the thread’s activity. 
        The method run() in <ants> representing the thread’s activity will be called. 
        Multiple threads are runing at the same time.

            @return self.shortest_path: The shortest path found.
            @return self.shortest_dist: The corresponding distance.
        """

        if path and os.path.exists(path):
            raise IOError('path already exists. please choose another to save history.')

        memory = np.full((self.iter, 2), None)

        for i in range(self.iter):
            self.ants = self.init_ants()
            
            for ant in self.ants:
                ant.start()

            for ant in self.ants:
                ant.join()
            
            for ant in self.ants:
                if ant.ended_before_goal: # ant got stuck
                    continue

                if not self.best_ant:
                    self.shortest_dist = ant.distance_traveled
                    self.shortest_path = ant.path
                    self.best_ant = ant

                if ant.distance_traveled < self.shortest_dist:
                    self.shortest_path = ant.path
                    self.shortest_dist = ant.distance_traveled
                    self.best_ant = ant

            #saving history
            memory[i, 0] = self.shortest_dist
            memory[i, 1] = self.shortest_path
            if path:
                np.save(path, memory)

            if i % 100 == 0:
                print('This is: Algorithm: ', self.algo, ' --- #Ants: ', self.ants_total, ' --- #Iterations: ', self.iter)
                print('alpha: ', self.alpha, 'beta: ', self.beta, 'rho: ', self.rho, 'q0: ', self.q0, 'init: ', self.init_pher)
                print('iteration', i, ':', 'shortest distance =', self.shortest_dist)

            self.update_pheromone()
        
        return self.shortest_path, self.shortest_dist, memory

    def update_pheromone(self):
        """
            Updates the pheromone graph, based on the ants movements.
            Several algorithms, such as ant system, elitist ant etc. are possible.
            The all comprise two steps: 1) Evaporation, 2) New pheromone based on the paths.
        """
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['pher'] *= (1 - self.rho)
            
        if self.algo == 'ant_system':
            for ant in self.ants:
                delta = ant.return_new_pher_trace()
                for edge in delta.edges():
                    self.graph[edge[0]][edge[1]]['pher'] += delta[edge[0]][edge[1]]['delta']

        if self.algo == 'elitist':
            if not self.init_pher:
                raise ValueError('must provide initial pheromone value')
            self.init_pheromone()
            delta = self.best_ant.return_new_pher_trace()
            pheromones = []
            for edge in delta.edges():
                self.graph[edge[0]][edge[1]]['pher'] += delta[edge[0]][edge[1]]['delta']
                pheromones.append(delta[edge[0]][edge[1]]['delta'])
            #print('mean pheromone:', np.mean(np.array(pheromones)))
            
        if self.algo == 'min_max':
            if not self.init_pher:
                raise ValueError('must provide initial pheromone value')
            if not self.min_pher or not self.max_pher:
                raise ValueError('must provide min and max values for pheromone')
            self.init_pheromone()
            delta = self.best_ant.return_new_pher_trace()
            pheromones = []
            for edge in delta.edges():
                new_pher = max(self.min_pher, min(delta[edge[0]][edge[1]]['delta'], self.max_pher))
                self.graph[edge[0]][edge[1]]['pher'] += new_pher
                pheromones.append(new_pher)
            #print('mean pheromone:', np.mean(np.array(pheromones)))

        if self.algo == 'ACS':
            #no initial pheromone values needed here (?)
            best_track = self.best_ant.return_new_pher_trace()
            for edge in best_track.edges():
                self.graph[edge[0]][edge[1]]['pher'] += 1 / self.shortest_dist
