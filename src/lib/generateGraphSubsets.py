#!/usr/bin/env python3

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from tabulate import tabulate



def generateGraphSubsets(G):
    """
    This function should create all possible subsets of nodes and return a list of tuples where each element of the
    outer list is a subset and each element of the inner tupple is a node belonging to said subset

    [(n1,n2...nn),(n2,n5...nn)]

    Considerations:
        -In this case the graphs nodes are scalars, yet they could be any object, I should be careful about that
    """

    assert isinstance(
        G, nx.Graph
    ), f"Input should be a networkX undirected graph object ({type(nx.Graph())}\n Input type was {type(G)}"

    nodes = list(G.nodes())
    subsets = []

    for i in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, i):
            subsets.append(subset)

    return subsets


def main():

    # When executed in main this line doens't make sense
    from randomGraphGenerators import generateRandomUndirectedGraph

    random.seed(98374)
    np.random.seed(98374)

    G = generateRandomUndirectedGraph(5, 0.6)
    generateGraphSubsets(G)

    pos = nx.get_node_attributes(G, "pos")

    nx.draw(G, pos, with_labels=True, node_color="skyblue")
    plt.show()


if __name__ == "__main__":
    main()
