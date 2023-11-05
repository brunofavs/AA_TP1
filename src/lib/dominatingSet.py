#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

def computeDominatingSet(G,subsets):
    """
    This function should take in a graph and its subsets and exclude the non-dominant sets and return a list of
    dominant sets
    """

    assert isinstance(
        G, nx.Graph
    ), f"Input should be a networkX undirected graph object ({type(nx.Graph())}\n Input type was {type(G)}"

    dominating_sets = []

    for subset in subsets:
        
        outsider_nodes = [node for node in G.nodes() if node not in subset]

        for outsider_node in outsider_nodes:
            # outsider
            pass
        # TODO Get outsider neighbours, check if they are in subset, if not its not a subset
            





    return dominating_sets




def main():
    pass

if __name__=="__main__":
    main()
