import sys
import os
import time
import argparse
import random
from progress import Progress


class Graph:
    def __init__(self, representation="adjacency_list"):
        """Initialise graph object"""
        self.representation = representation
        self.graph = None
        self.nodes = None
        self.weights = None
    def load_graph(self, datalife, weights=None):
        """Load graph from text file"""
        self.graph = {}
        self.weights = weights

        for line in datafile:
            source, target = line.strip().split()
            if source not in self.graph:
                self.graph[source] = []
            self.graph[source].append(target)
            if target not in self.graph:
                self.graph[target] = []
        
        # Conversion of graph to desired representation
        if self.representation == "adjacency_list":
            return self.graph
        elif self.representation == "weighted_adjacency_list":
            if weights is None:
                raise ValueError("Weights are required for weighted adjacency list.")
            return self.to_weighted_adjacency_list()
        elif self.representation == "matrix":
            return self.to_matrix()
        elif self.representation == "edge_list":
            return self.to_edge_list()
        else:
            raise ValueError(f"Unknown representation type: {self.representation}")
        
    def print_stats(self):
        """ Print number of nodes and edges in the given graph"""
        if isinstance(self.graph, dict):  # Adjacency list
            num_nodes = len(self.graph)
            num_edges = sum(len(targets) for targets in self.graph.values())
        elif isinstance(self.graph, list) and isinstance(self.graph[0], tuple):  # Edge list
            num_nodes = len({node for edge in self.graph for node in edge})
            num_edges = len(self.graph)
        elif isinstance(self.graph, tuple):  # Matrix
            matrix, nodes = self.graph
            num_nodes = len(nodes)
            num_edges = sum(sum(row) for row in matrix)
        else:
            raise ValueError("Unknown graph representation type.")

        print(f"Graph contains {num_nodes} nodes and {num_edges} edges.")

def to_edge_list(self):
        """
        Conversion of the default graph to an edge list representation.
        """
        edge_list = []
        for source, targets in graph.items():
            for target in targets:
                edge_list.append((source, target))

        return edge_list









