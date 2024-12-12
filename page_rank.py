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
    def load_graph(self, datafile, weights=None):
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
        for source, targets in self.graph.items():
            for target in targets:
                edge_list.append((source, target))

        return edge_list

    def to_matrix(self):
        """Converting the graph to an adjacency matrix representation."""
        self.nodes = list(self.graph.keys())
        index_map = {node: i for i, node in enumerate(self.nodes)}
        size = len(self.nodes)
        matrix = [[0] * size for _ in range(size)]

        for source, targets in self.graph.items():
            for target in targets:
                matrix[index_map[source]][index_map[target]] = 1

        return matrix, self.nodes

    def to_weighted_adjacency_list(self):
        """ Convert the graph to a weighted adjacency list."""
        weighted_graph = {
            source: {target: self.weights.get((source, target), 1) for target in targets}
            for source, targets in self.graph.items()
        }
        return weighted_graph

def stochastic_page_rank(graph, n_repetitions):
    """
    Stochastic PageRank estimation using adjacency list.

    Parameters:
        graph (dict): Adjacency list representation of the graph.
        n_repetitions (int): Number of repetitions for random walker.

    Returns:
        dict: Dictionary mapping nodes to their hit frequency.
    """
    hit_count = {node: 0 for node in graph}
    current_node = random.choice(list(graph.keys()))
    hit_count[current_node] += 1

    for _ in range(n_repetitions):
        if not graph[current_node]:  # No outgoing edges
            current_node = random.choice(list(graph.keys()))
        else:
            current_node = random.choice(graph[current_node])
        hit_count[current_node] += 1

    return hit_count

def distribution_page_rank(graph, n_steps):
    """
    Distribution-based PageRank estimation using adjacency list.

    Parameters:
        graph (dict): Adjacency list representation of the graph.
        n_steps (int): Number of iterations for convergence.

    Returns:
        dict: Dictionary mapping nodes to their PageRank scores.
    """
    num_nodes = len(graph)
    node_prob = {node: 1 / num_nodes for node in graph}  # Initial uniform probability

    for _ in range(n_steps):
        next_prob = {node: 0 for node in graph}

        for node, edges in graph.items():
            if edges:  # Nodes with outgoing edges
                p = node_prob[node] / len(edges)
                for target in edges:
                    next_prob[target] += p
            else:  # Nodes without outgoing edges (distribute uniformly)
                p = node_prob[node] / num_nodes
                for target in graph:
                    next_prob[target] += p

        node_prob = next_prob

    return node_prob


def stochastic_page_rank_edge_list(edge_list, n_steps, nodes):
    """Random walker estimation using edge list representation."""
    hit_count = {node: 0 for node in nodes}
    current_node = random.choice(nodes)
    hit_count[current_node] += 1

    for _ in range(n_steps):
        outgoing_edges = [(source, target) for source, target in edge_list if source == current_node]
        if not outgoing_edges:
            current_node = random.choice(nodes)
        else:
            _, current_node = random.choice(outgoing_edges)
        hit_count[current_node] += 1
    return hit_count


def distribution_page_rank_matrix(matrix, nodes, n_steps):
    """PageRank using adjacency matrix representation."""
    num_nodes = len(nodes)
    node_prob = [1 / num_nodes] * num_nodes
    for _ in range(n_steps):
        next_prob = [0] * num_nodes
        for i in range(num_nodes):
            row_sum = sum(matrix[i])
            if row_sum == 0:
                for j in range(num_nodes):
                    next_prob[j] += node_prob[i] / num_nodes
            else:
                for j in range(num_nodes):
                    if matrix[i][j] == 1:
                        next_prob[j] += node_prob[i] / row_sum
        node_prob = next_prob
    return {nodes[i]: prob for i, prob in enumerate(node_prob)}

parser = argparse.ArgumentParser(description="Estimates PageRanks from link information.")
parser.add_argument("datafile", nargs="?", type=argparse.FileType("r"), default=sys.stdin,
                    help="Text file of links among web pages.")
parser.add_argument("-m", "--method", choices=["stochastic", "distribution"], default="stochastic",
                    help="Selected PageRank algorithm.")
parser.add_argument("-r", "--repeats", type=int, default=1_000_000, help="Number of repetitions for stochastic method.")
parser.add_argument("-s", "--steps", type=int, default=100, help="Number of steps for distribution method.")
parser.add_argument("-n", "--number", type=int, default=20, help="Number of results to display.")
parser.add_argument("-g", "--representation", choices=["adjacency_list", "edge_list", "matrix", "weighted_adjacency_list"],
                    default="adjacency_list", help="Graph representation to use.")

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize the Graph class
    graph = Graph(representation=args.representation)

    # Load the graph
    graph.load_graph(args.datafile, weights={
        ("A", "B"): 3,
        ("A", "C"): 1,
        ("B", "C"): 2,
        ("C", "A"): 4
    } if args.representation == "weighted_adjacency_list" else None)

    # Print graph statistics
    graph.print_stats()

    start = time.time()

    # Handle PageRank based on representation
    if args.representation == "edge_list" and args.method == "stochastic":
        edge_list = graph.to_edge_list()
        nodes = list(set([node for edge in edge_list for node in edge]))
        ranking = stochastic_page_rank_edge_list(edge_list, args.repeats, nodes)

    elif args.representation == "matrix" and args.method == "distribution":
        matrix, nodes = graph.to_matrix()
        ranking = distribution_page_rank_matrix(matrix, nodes, args.steps)

    elif args.representation == "weighted_adjacency_list":
        ranking = graph.to_weighted_adjacency_list()
        ranking = sorted(ranking.items(), key=lambda item: sum(item[1].values()), reverse=True)

    else:  # Default adjacency list logic
        algorithm = distribution_page_rank if args.method == 'distribution' else stochastic_page_rank
        ranking = algorithm(graph.graph, args.steps if args.method == 'distribution' else args.repeats)

    stop = time.time()
    time_taken = stop - start

    # Output results
    if isinstance(ranking, dict):
        top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:args.number]
        print(f"Top {args.number} pages:")
        for node, value in top:
            print(f"{node}: {value:.6f}")
    elif isinstance(ranking, list):  # Handle weighted adjacency list case
        top = ranking[:args.number]
        print(f"Top {args.number} pages by total weight:")
        for node, weight_dict in top:
            total_weight = sum(weight_dict.values())
            print(f"{node}: {total_weight:.6f}")
    else:
        raise TypeError(f"Unexpected ranking type: {type(ranking)}")
    
    # Print calculation time
    print(f"Calculation took {time_taken:.2f} seconds.")






