import networkx as nx
import numpy as np
import threading


""" 
    Ant Colony Optimisation
    Lucas-Raphael Müller 
"""

class Ant(threading.Thread):
    """A class representing an ant as part of the ant colony.

    :param graph: A graph representation of the ant's world.
    :param init_loc: Initial location of the ant (usually placed randomly).
    :param alpha: Power of the pheromone factor.
    :param beta: Power of the attractiveness factor.
    :param unique_visit: Determines whether a node can only be visited once.
    :param goal: Determines the goal of the optimisation. Could be TSP, i.e. must visit all cities at least once.
    :param end_node: specifies the end_node for path minimisation
    """

    def __init__(self, colony, graph, init_loc, alpha=1, beta=1, unique_visit=False, goal='TSP', end_node=None):

        self.graph = graph
        self.init_loc = init_loc
        self.loc = init_loc
        self.alpha = alpha
        self.beta = beta
        self.unique_visit = unique_visit
        self.goal = goal
        self.end_node = end_node
        self.colony = colony

        self.complete = False
        self.ended_before_goal = False
        self.last_move = False
        self.path = []
        self.distance_traveled = 0.0
        self.poss_loc = list(self.graph.nodes())
        self.nodes = list(self.graph.nodes())

        threading.Thread.__init__(self)

        self.travel(self.init_loc, init=True)  # recognise initial location as part of the path

    def run(self):
        """Actual run of the ants. This method overloads threading.Thread.run"""

        while not self.is_goal_achieved(self.goal):
            possible_nodes = self.get_possible_nodes()
            if possible_nodes.__len__() < 1:
                self.ended_before_goal = True
                break
            next = self.return_best_node(self.loc, possible_nodes, self.alpha, self.beta)
            self.travel(next, init=False)

        self.complete = True

    def get_possible_nodes(self):
        """Prevent to get another node after path is closed."""
        if self.path.__len__() > 1 and self.path[0] == self.path[-1]:
            raise ValueError('Path is already closed. This should not happen.')

        """Returns nodes which are accessible (i.e. neighbors), and possible (i.e. may not be visited already). """
        nodes_all = np.intersect1d(list(self.graph.neighbors(self.loc)), self.poss_loc)

        """Do not allow for self loops."""
        nodes_not_self = np.setdiff1d(nodes_all, self.loc)

        """Allow to move to initial node for TSP even though initial node was already visited, 
        this must not occur more than once.
        """
        if self.path.__len__() == self.nodes.__len__() \
                and self.init_loc in list(self.graph.neighbors(self.loc)) \
                and self.unique_visit \
                and self.goal == 'TSP':
            self.poss_loc = []  # safety measure
            self.last_move = True
            return [self.init_loc]

        return nodes_not_self

    def return_best_node(self, current_node, possible_nodes, alpha, beta, heuristic=1):
        """Returns the best node out of the possible based on current pheromone level and distance to next node."""

        p = np.zeros(possible_nodes.__len__())
        p_sum = 0
        for i in range(p.__len__()):
            tau = self.graph[current_node][possible_nodes[i]]['pher']
            eta = 1 / self.graph[current_node][possible_nodes[i]]['weight']
            p[i] = tau ** alpha * eta ** beta
            p_sum += p[i]

        if p_sum > 0:
            p /= p_sum
        else:
            p = np.ones_like(p) / p.__len__()

        if self.colony.algo == 'ACS':
            q = np.random.uniform(0, 1)
            if q <= self.colony.q0:
                next = np.argmax(p)
                return possible_nodes[next]

        return np.random.choice(possible_nodes, p=p)

    def is_goal_achieved(self, goal):
        """Test whether goal is achieved."""
        if goal == 'TSP':
            if np.array_equal(np.unique(self.path), self.nodes) and self.loc == self.init_loc:
                return True
            else:
                return False
        elif goal == 'PathMin':
            if self.loc == self.end_node:
                return True
            else:
                return False
        else:
            raise ValueError('Only TSP possible so far.')

    def travel(self, next, init=False):
        """Updates path and distance."""

        self.path.append(next)
        if self.unique_visit and not self.last_move:  # last move to close the line; already removed.
            self.poss_loc.remove(next)

        if init is False:
            self.distance_traveled += self.graph[self.loc][next]['weight']

        #local updates after every step
        if self.colony.algo == 'ACS' or self.colony.algo == 'biological':
            self.graph[self.loc][next]['pher'] *= (1 - self.colony.rho)
            self.graph[self.loc][next]['pher'] += self.colony.rho * self.colony.tau

        self.loc = next

    def return_new_pher_trace(self):
        """ Returns a helper graph which feature the pheromone trace of a single ant denoted as 'delta'."""
        if self.complete:
            delta_graph = self.graph.copy()
            nx.set_edge_attributes(delta_graph, 0, 'delta')
            for i in range(self.path.__len__() - 1):
                delta_graph[self.path[i]][self.path[i + 1]]['delta'] = 1 / delta_graph[self.path[i]][self.path[i + 1]][
                    'weight']

            return delta_graph
        else:
            raise ValueError('Ant has not yet completed.')


class Ant_Colony:
    """A class representing an ant colony.

    :param graph: A graph representation of the world.
    :param ants_total: Total numbe of ants.
    :param iter: Number of iterations.
    :param alpha: Power of the pheromone factor.
    :param beta: Power of the attractiveness factor.
    :param rho: Determines the evaporation factor.
    :param unique_visit: Determines whether a node can only be visited once.
    :param goal: Determines the goal of the optimisation. Could be TSP, i.e. must visit all cities at least once.
    :param end_node: specifies the end_node for path minimisation
    """

    def __init__(self, graph, ants_total, iter, alpha, beta, rho, unique_visit, goal, start_node=None, end_node=None, \
                 init_pher=0.1, min_pher=None, max_pher=None, q0=0.3, tau=0.00001, algo='biological'):

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
        """Initialize array of ants."""
              
        ants = [None] * self.ants_total
        for i in range(self.ants_total):
            """For TSP ants are placed randomly, for shortest path a starting node needs to be specified beforehand.
            """
            if self.start_node is None and self.goal == 'TSP':
                self.start_node = np.random.choice(self.graph.nodes())
                
            ants[i] = Ant(colony=self, graph=self.graph, init_loc=self.start_node, alpha=self.alpha, beta=self.beta, \
                          unique_visit=self.unique_visit, goal=self.goal, end_node=self.end_node)
            
        return ants

    def init_pheromon(self):
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['pher'] = self.init_pher

    def find(self):
        """Start the thread’s activity. 
        The method run() in <ants> representing the thread’s activity will be called. 
        Multiple threads are runing at the same time.
        """
        for i in range(self.iter):
            self.ants = self.init_ants()
            
            for ant in self.ants:
                ant.start()

            for ant in self.ants:
                ant.join()
            
            for ant in self.ants:
                if ant.ended_before_goal: # ant got stuck but did meet goal
                    continue

            if not self.best_ant:
                self.shortest_dist = ant.distance_traveled
                self.shortest_path = ant.path
                self.best_ant = ant

            if ant.distance_traveled < self.shortest_dist:
                self.shortest_path = ant.path
                self.shortest_dist = ant.distance_traveled
                self.best_ant = ant

            print('iteration', i, ':', 'shortest distance =', self.shortest_dist)

            self.update_pheromon()
        
        return self.shortest_path, self.shortest_dist 
        
    def update_pheromon(self):
        """Updates the pheromon graph, based on the ants movements.
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
            self.init_pheromon()
            delta = self.best_ant.return_new_pher_trace()
            pheromones = []
            for edge in delta.edges():
                self.graph[edge[0]][edge[1]]['pher'] += delta[edge[0]][edge[1]]['delta']
                pheromones.append(delta[edge[0]][edge[1]]['delta'])
            print('mean pheromone:', np.mean(np.array(pheromones)))
            
        if self.algo == 'min_max':
            if not self.init_pher:
                raise ValueError('must provide initial pheromone value')
            if not self.min_pher or not self.max_pher:
                raise ValueError('must provide min and max values for pheromone')
            self.init_pheromon()
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