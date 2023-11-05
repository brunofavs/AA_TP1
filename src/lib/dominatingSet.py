#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

def computeDominatingSet(G,subset):
    """
    This function should take in a graph and its subset and determine if it is a dominant set
    """

    assert isinstance(
        G, nx.Graph
    ), f"Input should be a networkX undirected graph object ({type(nx.Graph())}\n Input type was {type(G)}"

    print(f'\n\nAnalyzing if subset {subset} is dominant')

    outsider_nodes = [node for node in G.nodes() if node not in subset]

    for outsider_node in outsider_nodes:
        outsider_neighbours = list(G.neighbors(outsider_node))

        print(f'Neighbours of outsider node {outsider_node} are {outsider_neighbours}')
    
        '''
        Loops through every node in subset
        Checks if it is in outsider neighbours
        Adds either True or False to the list
        If the list has atleast 1 True intersection returns True otherwise returns False
        '''

        # * Could use either approach
        intersection = any(node in outsider_neighbours for node in subset)
        intersection_list = [node for node in outsider_neighbours if node in subset]

        print(f"Outsider node neighbours {intersection_list} are also subset's nodes")

        # If there isn't a single outsider node neighbour also a subset node then its not dominant
        if not intersection:
            print(f'Neither neighbour of the outsider node {outsider_node} is in subset {subset}, this is NOT a dominating set')
            return False

    print(f'The subset {subset} is dominant')
    return True
        









def main():
    pass

if __name__=="__main__":
    main()
