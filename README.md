# aco
[![Build Status](https://travis-ci.org/Haydnspass/aco.svg?branch=master)](https://travis-ci.org/Haydnspass/aco)
Ant Colony Optimisation

## Execution
Specify parameters in run_aco.py either by modifying the file or by passing arguments via the console.
(python run_aco.py 'graph_file.txt' 'outputname' 'algorithm' 'TSP' ants_total iterations beta_value rho_value init_pher_value q0_value rho_local_value unique_visit_bool)

## Graphs
Put .txt file of graphs in data (if interdistance matrix provided) and in data/coordinates if
coordinates are provided.

## TSP
Travelling Salesman Problem
Test data taken from: https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
