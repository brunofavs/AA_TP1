#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate


def generateRandomUndirectedGraph(num_nodes, graph_density):
    '''
    Undirected graph density is D = 2*E / (n*(n-1))

    Doubts :
        -Does it make sense for only having 1 connected component?
            -> I dont think so, isn't it just like analysing 2 separate graphs?

        -Does it make sense to have isolated nodes?
            -> No because any set containing only isolated nodes will never be a dominating set and they can be including into any other dominating set.
               However, since there is no negative weights, they would only make the set's total weight bigger.
        
        -For graphs with few nodes, sometimes just assuring there is only 1 connected component already makes it denser that intended, I suppose
        there isn't anything I could do about that, right?
    '''

    assert graph_density <= 1 and graph_density >= 0,"Graph density must within [0,1]"
    
    num_edges = int(graph_density * num_nodes * (num_nodes - 1) / 2)

    G = nx.Graph()

    # Generate random (x, y) coordinates for each node
    coordinates = {node: (np.random.randint(0, 101), np.random.randint(0, 101)) for node in range(num_nodes)}
    # Generate weights 
    #!(nx only has built-in edge weight)
    
    weights = {node: (np.random.randint(0, 101)) for node in range(num_nodes)}

    for node in range(num_nodes):
        G.add_node(node, pos=coordinates[node],weight = weights[node])  # Add node with coordinates

    # Step 1: Generate a random connected graph using BFS
    visited = set()
    start_node = 0  # Start with node 0
    visited.add(start_node)

    """
    - A visited node is chosen
    - A non visited node is chosen as neighbour
    - The edge is added
    - The neighbour is visited
    - Cycle continues until every node is visited    
    """

    while len(visited) < num_nodes:
        current_node = np.random.choice(list(visited))
        neighbor = np.random.choice([node for node in range(num_nodes) if node not in visited])
        G.add_edge(current_node, neighbor)
        visited.add(neighbor)

    # Step 2: Add additional edges to achieve the desired density while maintaining connectivity

    while G.number_of_edges() < num_edges:
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)

        if node1 != node2 and not G.has_edge(node1, node2):
            G.add_edge(node1, node2)

    return G





def main():
    pass

if __name__=="__main__":
    main()
