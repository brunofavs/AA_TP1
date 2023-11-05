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
class LWDS:
    def __init__(self, n_nodes, density):
        self.n_nodes = n_nodes
        self.ideal_density = density
        self.graph = generateRandomUndirectedGraph(n_nodes, density)
        self.dominant_sets = []
        self.non_dominant_sets = []
        self.lightest_dominating_sets = []

        print(
            f"Generating a random undirected graph with {n_nodes} nodes and {density*100}% density"
        )
        print(f"Actual density of graph is {nx.density(self.graph)*100}%")

    def computeAllSubsets(self):
        # ! For big graphs this crashes the PC
        self.graph_subsets = generateGraphSubsets(self.graph)

    def computeDominantSets(self):
        for subset in self.graph_subsets:
            is_dominant = computeDominatingSet(self.graph, subset)

            if is_dominant:
                self.dominant_sets.append(subset)
            else:
                self.non_dominant_sets.append(subset)

        print(f"All dominant sets are {self.dominant_sets}")
        print(f"\n\nAll NON dominant sets are {self.non_dominant_sets}")

    def findLightestSets(self, subsets):
        all_weights = nx.get_node_attributes(self.graph, "weight")
        subsets_weights = np.array([])

        for subset in subsets:
            subset_weight = sum([all_weights[node] for node in subset])
            print(f"Subset's {subset} weigth is {subset_weight}")

            subsets_weights = np.append(subsets_weights, subset_weight)

        # This should return all possible minimum sets, and not just one
        minimum_idxs = np.where(subsets_weights == subsets_weights.min())

        # The [0] is necessary because np.where outputs a tuple of arrays
        print(f"The lightest dominant sets are : ")
        for idx in minimum_idxs[0]:
            self.lightest_dominating_sets.append(subsets[idx])
            print(f"    -> {subsets[idx]},weigth = {subsets_weights[idx]}")

    def bruteForceDominantSets(self):
        self.computeAllSubsets()
        self.computeDominantSets()
        self.findLightestSets(self.dominant_sets)

    def draw(self):
        # * Get the positions from the node's attributes
        pos = nx.get_node_attributes(self.graph, "pos")
        weights = nx.get_node_attributes(self.graph, "weight")

        labels = {
            node: f"""#{node}
w: {self.graph.nodes[node]['weight']}
{self.graph.nodes[node]['pos']}"""
            for node in self.graph.nodes
        }

        # * Draw the graph with coordinates
        nx.draw(
            self.graph,
            pos,
            labels=labels,
            with_labels=True,
            node_color="skyblue",
            font_size=12,
            node_size=2500,
        )
        plt.show()


def main():
    config = {"rng_seed": 98374}
    random.seed(config["rng_seed"])
    np.random.seed(config["rng_seed"])

    n_nodes = 30
    density = 1

    G = LWDS(n_nodes, density)
    G.bruteForceDominantSets()
    G.draw()


if __name__ == "__main__":
    main()
