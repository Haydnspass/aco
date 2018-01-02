import threading
import random
import math

"""
	Jacqueline Wagner.
"""

class ants_colony:
	"""
		A colony of ants is initializied. The goal is to find an optimal path through a map by making use of ants pheromone distribution.
	"""

	class ant(threading.Thread):
		"""
			This class createes an ant which can later be used in the ant_colony class.
			"""
		def __init__(self, init_loc, first_time, poss_loc, pher_map, distance, alpha, beta):
			"""
				Initializing an ant.
			
				@param init_loc - the location in which the ant starts its traversal.
				@param first_time - indicates whether this is the first time traversing the map.
				@param poss_loc - the possible location to which the ant can move to. Dosen't include the locations already traveled to.
				@param pher_map - the map updated with pheromone values for each path between two nodes.
				@param distance - a function which calculates the distance between two nodes.
				@param alpha - regulates the influence the amount of pheromone has on the decision to pick certain paths.
				@param beta - regulates the influence the distance to the next node has on the decision to pick certain paths.
		
				@value loc - the location in which the ant is currently.
				@value complete - marks wether the ant has completed its journey or not.
				@value path - a list containing information about the nodes the ant has been to.
				@value pher_trail - a list containing the pheromone values the ant deposited along its path.
				@value distance_traveled - total distance traveled by the ant over its entire path.
				"""
		
			"""
				Initializing the ant.
			"""
			self.init_loc = init_loc
			self.loc = init_loc
			self.first_time = first_time
			self.poss_loc = poss_loc
			self.complete = False
			self.path = []
			self.distance_traveled = 0.0
			self.pher_map = pher_map
			self.distance = distance
			self.alpha = alpha
			self.beta = beta
			threading.Thread.__init__(self)

			"""
				The starting location is added to the path before the traversal begins.
			"""
				self.update_path(init_loc)

		def run(self):
			"""
				While there are still possible locations to travel to, a next node is found and the ant travels there.
			"""
			while self.poss_loc.size > 0:
				next = self.next_node()
				self.travel(self.loc, next)
			
			"""
				There are no more possible locations to travel to. The journey of the ant is complete.
			"""
			self.complete = True
	
		def travel(self, current, next):
			"""
				the path and the distance traveld are updated and will then include the new part of the path from next_node.
				The new location is updated to show the new location from next_node
			
				@param current - current position of the ant in the map.
				@param next - position the ant is traveling to.
			"""
			self.update_path(next)
			self.update_distance(current, next)
			self.location = next
	
		def update_path(self, next):
			"""
				The next node is added to the path and subsequently removed from the list of possible locations.
				
				@param next - Position the ant has traveled to.
				"""
			self.path.append(next)
			self.poss_loc.remove(next)

		def update_distance(self, current, next):
			"""
				The distance traveld by the ant has to be updated after the path is updated.
				
				@param current - Position the ant was at previously.
				@param next - Position the ant has traveled to.
			"""
			self.distance_traveled += float(self.distance(current, next))
		
		def return_path(self):
			"""
				Obtain the path traveled by the ant once the map has been successfully traversed.
				If the traversal is not complete yet nothing is returned.
				
				@return path - total path traveled by the ant.
			"""
			if self.complete:
				return self.path
			return None
		
		def return_distance_traveled(self):
			"""
				Obtain the distance traveled by the ant once the map has been successfully traversed.
				If the traversal is not complete yet nothing is returned.
				
				@return distance_traveled - total distance traveled by the ant.
			"""
			if self.complete:
				return self.distance_traveled
			return None
		
		def next_node(self):
			"""
				The next node of the ant is selected.
				The benefit of each move from the current position to the next node is calculated. Then the next node is chosen randomly based on its benefit.
			"""
			
			"""
				If this is the first time traversing the map there is no pheromone distributed yet. The choice can be made completly at random.
			"""
			if self.first_time:
				return random.choice(self.pos_loc)
			
			benefit = dict()
			benefit_sum = 0.0
			""""
				The benefit of moving to each possible location is calculated based on the ACO algorithm.
				The sum of all benefits is calculated.
			"""
			for poss_next_loc in self.poss_loc:
				dist = float(self.distance(self.loc, poss_next_loc))
				pher_amount = float(self.pher_map[self.loc][poss_next_loc])
				benefit[poss_next_loc] = pow(pher_amount, self.alpha)*pow(1/dist, self.beta)
				benefit_sum += benefit[poss_next_loc]
			
			"""
				The next path from benefit is chosen at random.
				https://github.com/pjmattingly/ant-colony-optimization/blob/master/ant_colony.py
			"""
			make_decision = random.random()
			add = 0
			for poss_next_loc in benefit:
				weight = (benefit[poss_next_loc] / benefit_sum)
				if make_decision <= weight + add:
					return poss_next_loc
				add += weight
			

	def __init__(self, nodes, ants_total, distance, iter, pher_dep, pher_adjust, start=None, alpha, beta):
		"""
			An ant colony is initialized using ants from the ant class.
		
			@param nodes - Conects nodes to its corresponding values. This is a dict.
			@param ants_total - amount of ants traversing the map.
			@param distance - Calculates the distance between two nodes.
			@param iter - Defines how many times the ants will traverse the map.
			@param pher_dep - used to deposit pheromone values.
			@param pher_adjust - used to adjust pheromone values.
			@param start - Node where all ants begin their traversal.
			@param alpha - regulates the influence the amount of pheromone has on the decision to pick certain paths.
			@param beta - regulates the influence the distance to the next node has on the decision to pick certain paths.
	
			@value ants - Contains the ants traversing the map.
			@value all_distances - values of distances calculated between nodes.
			@value revised_pher_map - Contains the final values of pheromone used to calculate the next nodes. Theses values were adjusted based on pher_ and pher_evap.
			@value current_pher_map - Contains the values of pheromone deposited by the ants during the current traversal. Once theses values have been adjusted they are added to the revised_pher_map. It is then reset for the next traversal.
			@values first_time - indicates whether this is the first time traversing the map. This changes a few things since there is no pheromone deposition yet.
			@shortest_dist - the shortest distance in any ant traversal.
			@shortetst_path - the shortest path corresponding to the shortest_dist
		"""

		"""
			Initializing nodes.
		"""
		if len(nodes) < 1:
			raise ValueError("Nodes must contain at least one node.")
		
		if type(nodes) is not dict:
			raise TypeError("Nodes must be a dict.")
	
		"""
			Initializing ants_total.
		"""
		if ants_total < 1:
			raise ValueError("There must be at least one ant.")
				
		if type(ants_total) is not int:
			raise TypeError("The amount of ants must be an integer.")
					
		self.ants_total = ants_total
							
		"""
			Initializing distance method.
		"""
		if not callable(distance):
			raise TypeError("Distance has to be a method.")
									
		self.distance = distance
			
		"""
			Initializing the amount of iterations.
		"""
		if iter < 0:
			raise ValueError("There must be at least one iteration.")

		if (type(iter) is not int):
			raise TypeError("The amount of iterations must be an integer.")

		self.iter = iter
			
		"""
			Initializing the pher_dep parameter.
		"""
		if (type(pher_dep) is not int) and type(pher_dep) is not float:
			raise TypeError("The pher_dep parameter must be an integer or a float.")
					
		self.pher_dep = float(pher_dep)
			
		"""
			Initializing the pher_adjust parameter.
		"""
		if (type(pher_adjust) is not int) and type(pher_adjust) is not float:
				raise TypeError("The Pher_adjust parameter must be an integer or a float.")
					
		self.pher_adjust = float(pher_adjust)
						
		"""
			Initializing start.
		"""
		if start is None:
			self.start = 0
		else:
			self.start = None
			for key, val in self.nodes_keys.items():
				if val == start:
					self.start = key
			if self.start is None:
				raise KeyError("The start Key could not be found in nodes.")

		"""
			Create the ants for the colony.
		"""
		self.ants = self.init_ants(self.start)
				
		"""
			Initializing alpha.
		"""
		if alpha < 0:
			raise ValueError("Alpha must be larger or equal to zero.")
	
		if (type(alpha) is not int) and type(alpha) is not float:
			raise TypeError("Alpha must be an integer or a float.")

		self.alpha = float(alpha)
																											
		"""
			Initializing beta.
		"""
		if beta < 1:
			raise ValueError("Beta must be larger or equal to one.")

		if (type(beta) is not int) and type(beta) is not float:
			raise TypeError("Beta must be an integer or a float.")

		self.beta = float(beta)

		"""
			Create dict to map the keys to values of the nodes.
		"""
		self.nodes_keys, self.nodes = self.init_nodes(nodes)

		"""
			Create a matrix which contains the adjusted pher values for all the traversals.
		"""
		self.revised_pher_map = self._init_matrix(len(nodes), 0.0)

		"""
			Create a matrix which contains the deposited pher values for the current traversal.
		"""
		self.current_pher_map = self._init_matrix(len(nodes), 0.0)

		"""
			Create a matrix which contains distances between nodes that have already been calculated.
		"""
		self.all_distances = self.init_matrix(len(nodes), 0.0)

		"""
			Initialize to the first traversal.
		"""
		self.first_time = True

		"""
			Initialize the shortest distance and the corresponding shortest path.
		"""
		self.shortest_dist = None
		self.shortest_path = None

	def init_nodes(self, nodes):
		"""
			Creates two dicts. One asssigns the n nodes passed to numbers (0... n-1).
			The other one maps the numbers (0...n-1) to the corresponding value of each node.
			
			@param nodes - nodes passed.
			
			@return nodes_keys - Dict with keys to the nodes.
			@return nodes_values - Dict with values to the keys.
		"""
			nodes = dict()
			nodes_keys = dict()
			
			i = 0
			for key in sorted(nodes.keys()):
				nodes_keys[i] = key
				nodes[i] = nodes[key]
				i = i + 1
						
			return nodes_keys, nodes
	
	def init_ants(self, start):
		"""
			If this is the first time traversing the map, a number of ants are created.
			Otherwise the ants are reset to default conditions.
			
			@param start - Node where the ants will begin their traversal.
			
			@return ants - List of all ants.
		"""
		ants = []
		if self.first_time:
			for i in range self.ants_total:
				ants[i] = self.ant(start, first_time=True, self.nodes.keys(), self.revised_pher_map, self.path_distance, self.alpha, self.beta)
		else:
			i = 0
			for ant in self.ants:
				ants[i] = ant.__init__(start, first_time=False, self.nodes.keys(), self.revised_pher_map, self.path_distance, self.alpha, self.beta)
				i += 1
		return ants

	def init_matrix(self, size, val):
		"""
			Creates a n x n matrix.
			
			@param size - size n of matrix.
			@param val - all items of the matrix are set to this value.
			
			@return matrix - the n x n matrix with value val at each position.
		"""
		matrix=[]
		row=[]
		for i in range(size):
			for j in range(size):
				row.append(float(val))
			matrix.append(row)
		return matrix
	
	def path_distance(self, current, next):
		"""
			Calculates the distance between to nodes using the distance method.
			If this is the first time calculating the distance it is added to all_distances, otherwise it is returned from all_distances.
			
			@param current - node from which the distance is calculated.
			@param next - node to which the distance is calculated.
			
			@return all_distances - Values of all distances already calculated. Possibly updated by new value.
		"""
		if not self.all_distances[current][next]:
			distance = self.distance(self.nodes[current], self.nodes[next])
			if (type(distance) is not int) and (type(distance) is not float):
				raise TypeError("The return value of the distance method has to be an integer or a float.")
			self.all_distances[current][next] = float(distance)
			return distance
		return self.all_distances[current][next]

	def addto_current_pher_map(self, ant):
		"""
			The current_pher_map is updated with pheromone values along the ant's path.
			
			@param ant - the ant which path is evaluated.
		"""
		path = ant.get_path()
		for j in range(len(path)-1):
			previous_pher_value = float(self.current_pher_map[path[j]][path[j+1]])
			"""
				If the traversal isn't complete NONE is returned by return_distance_traveled.
				The new value has to be added to two spots in the matrix, j - j+1 and j+1 - j since these both represent the same path.
			"""
			updated_pher_value = self.pher_dep/ant.return_distance_traveled()
			new_pher_value = previous_pher_value + updated_pher_value
			self.current_pher_map[path[j]][path[j+1]] = new_pher_value
			self.current_pher_map[path[j+1]][path[j]] = new_pher_value

	def revise_pher_map(self):
		"""
		After all ants have traversed the map the unadjusted deposited pheromone values on the current_pher_map are revised and added to the revised_pher_map.
		"""
		for i in range(len(self.revised_pher_map)):
			for j in range(len(self.revised_pher_map)):
				self.revised_pher_map[i][j] = (1-self.pher_adjust)*self.revised_pher_map[i][j]
				self.revised_pher_map[i][j] += self.current_pher_map[start][end]

		
	def find(self):
		"""
			The ants traverse the map <iter>-times, the pheromone map is updated with the pheromone values.
			https://github.com/pjmattingly/ant-colony-optimization/blob/master/ant_colony.py
			
			@return path - shortest path.
		"""
		
		for _ in range(self.iter):
			"""
				Start the thread’s activity. The method run() in <ants> representing the thread’s activity will be called. Multiple threads are runing at the same time.
			"""
			for ant in self.ants:
				ant.start()
		
			"""
				Wait until the thread terminates before joining.
			"""
			for ant in self.ants:
				ant.join()
		
			for ant in self.ants:
				"""
					Updates the current_pher_map with this ants distribution of pheromone.
				"""
				self.addto_current_pher_map(ant)
				
				"""
					Initialize and update shortest_path and shortest_dist.
				"""
				if not self.shortest_path:
					self.shortest_path = ant.get_path()
				
				if not self.shortest_dist:
					self.shortest_dist = ant.return_distance_traveled()

				if ant.return_distance_traveled() < self.shortest_dist:
					self.shortest_path = ant.get_path()
					self.shortest_dist = ant.return_distance_traveled()
			
			"""
				The pheromone values are adjusted and therefore moved from current_pher_map to revisde_pher_map. Current_pher_map is reset. All ants are reset.
			"""
			self.revise_pher_map()
			self.current_pher_map = self.init_matrix(len(self.nodes), 0)
			self.init_ants(self.start)
			
			"""
				Any traversal after the first one has to be marked first_time = False.
			"""
			if self.first_time:
				self.first_time = False
		
		path = []
		for i in self.shortest_path:
			path.append(self.nodes_keys[i])

		return path

if __name__ == "__main__":
	nodes = {0: (0, 3), 1: (4, 8), 2: (10, 5), 3: (12, 10), 4: (9, 13)
	5: (13, 8), 6: (7, 12), 7: (13, 11), 8: (10, 11), 9: (11, 8)}

	def distance(current, next):
		x_distance = abs(current[0] - next[0])
		y_distance = abs(current[1] - next[1])
		return math.sqrt(pow(x_distance, 2) + pow(y_distance, 2))

	colony = ants_colony)(nodes, 50, distance, 70, pher_dep = 1000.0, pher_adjust = 0.5, alpha = 0.5, beta = 1.0)
	print(answer = colony.find())
