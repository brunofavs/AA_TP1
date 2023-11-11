#!/usr/bin/env python3

import os
import sys
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from tabulate import tabulate
import time
import csv

from lib.randomGraphGenerators import generateRandomUndirectedGraph
from lib.generateGraphSubsets import generateGraphSubsets
from lib.dominatingSet import testDominatingSet


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
            is_dominant = testDominatingSet(self.graph, subset)

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

    def greedy(self):
        """Order the nodes by the ratio neigbours/weight, choose the best until there is a dominating set"""
        pass
        node_list = np.asarray(self.graph.nodes)
        weight_list = [self.graph.nodes[n]["weight"] for n in self.graph.nodes]
        # Converting to array so that I can multiply by a scalar
        neighbour_list = np.asarray(
            [len(list(self.graph.neighbors(node))) for node in self.graph.nodes]
        )

        # Increasing influence of number of edges
        neighbour_list *= 1

        # * The lower the better
        weight_neighbours_ratio = np.divide(weight_list, neighbour_list)

        print(f"Node list {node_list}")
        print(f"Weights {weight_list}")
        print(f"Neighbours {neighbour_list}")
        print(f"Weight neighbour ratio is {weight_neighbours_ratio}")
        # Ordered from smallest to largest
        idx_ordered = np.argsort(weight_neighbours_ratio)
        print(f"Indexes ordered {idx_ordered}")

        # print(idx_ordered)
        node_list_ordered = list(node_list[idx_ordered])
        # print(node_list_ordered)

        is_dominant = False
        potential_dominant_set = []

        while is_dominant == False and len(node_list_ordered) != 0:
            potential_dominant_set.append(node_list_ordered.pop(0))

            is_dominant = testDominatingSet(self.graph, potential_dominant_set)

        # In case the while breaks because of len = 0
        if is_dominant == True:
            self.lightest_dominating_sets = potential_dominant_set

        weights = [
            self.graph.nodes[node]["weight"] for node in self.lightest_dominating_sets
        ]
        # weights = [node for node in self.lightest_dominating_sets]
        print(f"fewfwefew {weights}")
        subset_weight = sum(weights)

        print(f"Lightest dominant set is {self.lightest_dominating_sets}")
        print(f"Subset's {self.lightest_dominating_sets} weigth is {subset_weight}")

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

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def main():
    config = {"rng_seed": 98374}
    random.seed(config["rng_seed"])
    np.random.seed(config["rng_seed"])

    #* -----------------
    #* Single Testing
    #* -----------------
    # n_nodes = 8
    # # 18 is the limit for brute force
    # density = 0.4

    # G = LWDS(n_nodes, density)

    # start = time.time()
    # # G.greedy()
    # G.bruteForceDominantSets()
    # end = time.time()
    # print(end - start)
    # G.draw()
    # exit()

    #* -----------------
    #* Batch Testing
    #* -----------------
    n_nodes_list = [n for n in range(4, 200)]
    
    densities = [0.125, 0.25, 0.5, 0.75]
    csvs_bf_times = ["../results/bf_time_12_5.csv","../results/bf_time_25.csv","../results/bf_time_50.csv","../results/bf_time_75.csv"]
    csvs_gh_times = ["../results/gh_time_12_5.csv","../results/gh_time_25.csv","../results/gh_time_50.csv","../results/gh_time_75.csv"]
    csvs_n_ops= ["../results/bf_ops_12_5.csv","../results/bf_time_25.csv","../results/bf_time_50.csv","../results/bf_time_75.csv"] 

    times = []
    Gs = []

    # for csv_path,density in zip(csvs_bf_times,densities):
    for csv_path,density in zip(csvs_gh_times,densities):
        
        print(f'Calculating for density {density}')

        Gs = []
        times = []

        blockPrint()
        for n_nodes in n_nodes_list:
            Gs.append(LWDS(n_nodes,density))

        for G in Gs:
            start = time.time()
            # G.bruteForceDominantSets()
            G.greedy()
            end = time.time()

            times.append(float(f"{end-start:.2g}"))
        enablePrint()

        print(list(zip(n_nodes_list,times)))
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Writing header
            writer.writerow(['Nodes', 'Time'])

            # Writing data
            writer.writerows(zip(n_nodes_list, times))

if __name__ == "__main__":
    main()
