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
        
        # Convert graph to desired representation
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











