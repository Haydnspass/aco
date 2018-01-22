import networkx as nx
import numpy as np
import threading


""" 
    Jacqueline Wagner, Lucas Möller, Lucas-Raphael Müller 
"""

class Ant(threading.Thread):
    """
        A class representing an ant as part of the ant colony.
    """
    
    def __init__(self, colony, graph, init_loc, alpha=1, beta=1, unique_visit=False, goal='TSP', end_node=None):
    """
        Initializes an ant.
        
        @param colony: Instance of ant colony class.
        @param graph: A graph representation of the ant's world.
        @param init_loc: Initial location of the ant (usually placed randomly).
        @param alpha: Power of the pheromone factor.
        @param beta: Power of the attractiveness factor.
        @param unique_visit: Determines whether a node can only be visited once.
        @param goal: Determines the goal of the optimisation. Could be TSP, i.e. must visit all cities at least once.
        @param end_node: specifies the end_node for path minimisation
    """
        self.graph = graph
        self.init_loc = init_loc
        self.alpha = alpha
        self.beta = beta
        self.unique_visit = unique_visit
        self.goal = goal
        self.end_node = end_node
        self.colony = colony

        self.loc = None
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
        """
            Actual run of the ants. This method overloads threading.Thread.run
        """
        while not self.is_goal_achieved(self.goal):
            possible_nodes = self.get_possible_nodes()
            if possible_nodes.__len__() < 1:
                self.ended_before_goal = True
                break
            next = self.return_best_node(self.loc, possible_nodes, self.alpha, self.beta)
            self.travel(next, init=False)

        self.complete = True

    def get_possible_nodes(self):
        """
            Prevent to get another node after path is closed.
            
            @called in: run
            
            @return nodes_not_self: The nodes which are accessible, possible and don't result in a loop.
        """
        if self.path.__len__() > 1 and self.path[0] == self.path[-1]:
            raise ValueError('Path is already closed. This should not happen.')

        """
            Returns nodes which are accessible (i.e. neighbors), and possible (i.e. may not be visited already). 
        """
        nodes_all = np.intersect1d(list(self.graph.neighbors(self.loc)), self.poss_loc)

        """
            Do not allow for self loops.
        """
        nodes_not_self = np.setdiff1d(nodes_all, self.loc)

        """
            Allow to move to initial node for TSP even though initial node was already visited, 
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
        """
            Returns the best node out of the possible based on current pheromone level and distance to next node.
            
            @called in: run
            
            @return np.random.choice(possible_nodes, p=p)
            
            @param current_node: Node which the ant is currently located at.
            @param possible_nodes: All nodes which the ant can travel to.
            @param alpha: Power of the pheromone factor.
            @param beta: Power of the attractiveness factor.
            @param heuristic:
        """
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
        """
            Tests whether goal is achieved.
            
            @called in: run
            
            @return True or False
            
            @param goal: The goal which needs to be achieved.
        """
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
        """
            Updates path and distance.
            
            @called in: run, __init__
            
            @param next: Next node the ant will travel to.
            @param init: 
        """
        self.path.append(next)
        if self.unique_visit and not self.last_move:  # last move to close the line; already removed.
            self.poss_loc.remove(next)

        if init is False:
            self.distance_traveled += self.graph[self.loc][next]['weight']
            # local updates after every step
            if self.colony.algo == 'ACS' or self.colony.algo == 'biological':
                self.graph[self.loc][next]['pher'] *= (1 - self.colony.rho)
                self.graph[self.loc][next]['pher'] += self.colony.rho * self.colony.tau

        self.loc = next

    def return_new_pher_trace(self):
        """ 
            Returns a helper graph which feature the pheromone trace of a single ant denoted as 'delta'.       
        """
        if self.complete:
            delta_graph = self.graph.copy()
            nx.set_edge_attributes(delta_graph, 0, 'delta')
            for i in range(self.path.__len__() - 1):
                delta_graph[self.path[i]][self.path[i + 1]]['delta'] = 1 / delta_graph[self.path[i]][self.path[i + 1]][
                    'weight']

            return delta_graph
        else:
            raise ValueError('Ant has not yet completed.')
