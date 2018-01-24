#initialize an ant colony on a graph
for i in range(iterations):
    for ant in colony.ants: #potentially parallel
        while not ant.achieved_goal:
            ant.travel()
            ant.local_pheromone_update() #only in ACS
    #after all ants achieved goal
    colony.global_pheromone_update()