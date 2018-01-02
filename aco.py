import networkx as nx
import numpy as np
from itertools import enumerate

class Ant_Colony:
    class Ant:
        def __init__(self, graph, start, is_goal, distance_metric, alpha, beta, start=np.random.choice(nx.nodes(graph)), unique_visit=True):

            self.start = start
            self.current_node = start
            self.is_goal = is_goal
            self.unique_visit = unique_visit
            self.rest = False
            self.route = []
            self.dist = 0
            self.world = graph
            self.alpha = 1
            self.beta = 1
            
            
        def run_ant(self):
            while possible_locations(self.current_node):
                next = self.select_node(self.current_node, self.possible_locations(self.current_node), self.alpha, self.beta)
                self.update_state()
            
            self.rest = True
            
            
        def possible_nodes(self, node):
            if self.rest:
                return None
            if self.unique_visit:
                # all neighbors except already visited ones
                return np.setdiff1d(nx.all_neighbors(self.world, node), self.route)
            else:
                return nx.all_neighbors(self.world, node)
            
            
        def select_node(self, current_node, neighbor_nodes, alpha, beta, heuristic=1):
            pheromone = nx.get_edge_attributes(self.world, 'pheromone')
            
            p = np.zeros(neighbor_nodes.__len__())
            p_sum = 0
            for i in range(p.__len__()):
                p[i] = pheromone[(current_node, neighbor_nodes[i])]**alpha * heuristic**beta
                p_sum += p[i]
                
            p /= p_sum
            
            return np.random.choice(neighbor_nodes, p)
            
            
        def update_state(self, next_node):
            assert self.rest == False
            
            self.route.append(next_node)
            self.dist += 1
            if self.is_goal(next_node):
                self.rest = True
            self.current_node = next_node

            
    def update_pheromon(graph, ants):
        for i,ant in enumerate(ants):
            
