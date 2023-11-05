#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from tabulate import tabulate

from lib.randomGraphGenerators import generateRandomUndirectedGraph
from lib.generateGraphSubsets import generateGraphSubsets
from lib.dominatingSet import computeDominatingSet

# Lowest weight dominating set
class LWDS():
    def __init__(self,n_nodes,density):
        self.n_nodes = n_nodes
        self.ideal_density = density
        self.graph = generateRandomUndirectedGraph(n_nodes,density)

        print(f'Generating a random undirected graph with {n_nodes} nodes and {density*100}% density') 
        print(f'Actual density of graph is {nx.density(self.graph)*100}%')
    
    def computeSubsets(self):
        # ! For big graphs this crashes the PC
        self.graph_subsets = generateGraphSubsets(self.graph)
    
    def computeDominantSets(self):
        computeDominatingSet(self.graph,self.graph_subsets)

    def draw(self):

        #* Get the positions from the node's attributes
        pos = nx.get_node_attributes(self.graph, 'pos')
        weights = nx.get_node_attributes(self.graph,'weight')

        #* Draw the graph with coordinates
        nx.draw(self.graph, pos,labels = weights, with_labels=True, node_color='skyblue')
        # nx.draw(self.graph, pos, with_labels=True, node_color='skyblue')
        plt.show()

def main():
    config = {"rng_seed" : 98374}
    random.seed(config["rng_seed"])
    np.random.seed(config["rng_seed"])

    n_nodes = 15
    density = 0.5

    G = LWDS(n_nodes,density)
    G.computeSubsets()
    # G.computeDominantSets()
    G.draw()




if __name__=="__main__":
    main()
