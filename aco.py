import networkx as nx
import numpy as np
import threading

class Ant_Colony:
    class Ant(threading.Thread):
        def __init__(self, graph, init_loc, first_time, distance, alpha=1, beta=1, unique_visit=False, goal='TSP'):

            self.init_loc = init_loc
            self.loc = init_loc
            self.first_time = first_time
            self.complete = False
            self.path = []
            self.distance_traveled = 0.0
            self.distance = distance
            self.alpha = alpha
            self.beta = beta
            self.graph = graph
            self.poss_loc = list(self.graph.nodes())
            self.nodes = list(self.graph.nodes())
            self.unique_visit = unique_visit
            threading.Thread.__init__(self)

            # remove initial node as it was visited
            self.travel(None, self.init_loc)
            
            
        def run(self):
            while not self.is_goal_achieved():
                possible_nodes = self.get_possible_nodes()
                if possible_nodes.__len__() < 1:
                    break
                next = self.return_best_node(self.loc, possible_nodes, self.alpha, self.beta)
                self.travel(self.loc, next)

            self.complete = True

        def get_possible_nodes(self):
            return np.intersect1d(list(self.graph.neighbors(self.loc)), self.poss_loc)

        def is_goal_achieved(self, goal='TSP'):
            if goal == 'TSP':
                if np.array_equal(np.unique(self.path), self.nodes):
                    return True
                else:
                    return False
            else:
                raise ValueError('Only TSP possible so far.')
            
        def travel(self, current, next):
            # Update path
            self.path.append(next)
            if self.unique_visit:
                self.poss_loc.remove(next)
            
            # Update distance
            if current is not None:
                self.distance_traveled += self.graph[current][next]['weight']

        def return_new_pher_trace(self):
            if self.complete:
                delta_graph = self.graph.copy()
                nx.set_edge_attributes(delta_graph, 0, 'delta')
                for i in range(self.path.__len__() - 1):
                    delta_graph[self.path[i]][self.path[i + 1]]['delta'] = 1/delta_graph[self.path[i]][self.path[i + 1]]['weight']
                    
                return delta_graph
            else:
                raise ValueError('Ant has not yet completed.')

        def return_best_node(self, current_node, possible_nodes, alpha, beta, heuristic=1):

            p = np.zeros(possible_nodes.__len__())
            p_sum = 0
            for i in range(p.__len__()):
                p[i] = self.graph.get_edge_data(current_node,possible_nodes[i], default=0)['pher']**alpha * heuristic**beta
                p_sum += p[i]

            if p_sum > 0:
                p /= p_sum
            else:
                p = np.ones_like(p) / p.__len__()
            
            return np.random.choice(possible_nodes, p=p)

    def __init__(self, graph, ants_total, iter, alpha, beta):
        
        self.graph = graph
        self.ants_total = ants_total
        self.iter = iter
        self.alpha = alpha
        self.beta = beta
        #self.start = start
        self.shortest_dist = None
        self.shortest_path = None

        self.ants = self.init_ants()
        
        
    def init_ants(self, ants=None):
        if ants is None:
            ants = [None]*self.ants_total
            for i in range(self.ants_total):
                ants[i] = self.Ant(self.graph, np.random.choice(self.graph.nodes()), True, 1, self.alpha, self.beta, True, 'TSP')
        else:
            i = 0
            for ant in self.ants:
                ants[i] = ant.__init__(self.graph, np.random.choice(self.graph.nodes()), False, 1, self.alpha, self.beta, True, 'TSP')
                i += 1
        return ants
        
    def find(self):
        """
            Start the thread’s activity. The method run() in <ants> representing the thread’s activity will be called. Multiple threads are runing at the same time.
        """
        for ant in self.ants:
            ant.start()

        for ant in self.ants:
            ant.join()
        # for ant in self.ants:
        #     ant.run()
        
        best_ant = None
        for ant in self.ants:
            if not self.shortest_path:
                self.shortest_path = ant.path
                
            if not self.shortest_dist:
                self.shortest_dist = ant.distance_traveled
                
            if ant.distance_traveled < self.shortest_dist:
                self.shortest_path = ant.path
                self.shortest_dist = ant.distance_traveled
                best_ant = ant
            
        self.update_pheromon(self.ants)
        
        self.init_ants(self.ants)
        
        return self.shortest_path        
        
        
    def update_pheromon(self, ants, rho=0.1, best_ant=None, algorithm='ant_system'):
        # evaporation
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['pher'] *= (1 - rho)
            
        if algorithm == 'ant_system':
            for ant in self.ants:
                delta = ant.return_new_pher_trace()
                for edge in delta.edges():
                    self.graph[edge[0]][edge[1]]['pher'] += delta[edge[0]][edge[1]]['delta']
            
