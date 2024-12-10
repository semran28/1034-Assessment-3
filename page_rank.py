import sys
import os
import time
import argparse
import random
from progress import Progress


def load_graph(args):
    """Load graph from text file

    Parameters:
    args -- arguments named tuple

    Returns:
    A dict mapling a URL (str) to a list of target URLs (str).
    """
    graph={}

    # Iterate through the file line by line
    for line in args.datafile:
        # And split each line into two URLs   
        source, target =line.strip().split()
          
           # Add source to graph if it is not present
        if source not in graph:
            graph[source]=[]
            
        graph[source].append(target)
        # Ensures target URL also exists in dictioanry    
        if target not in graph:
                 graph[target]=[]
    
    return graph

def print_stats(graph):
    """Print number of nodes and edges in graph"""
    # Count nodes
    num_nodes = len(graph)
    # Count edges
    num_edges = sum(len(targets) for targets in graph.values())
    # Print the stats
    print(f"Graph contains {num_nodes} nodes and {num_edges} edges.")


def stochastic_page_rank(graph, args):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its hit frequency."""

    # Initial visit counts for each node:
    visit_counts= {node:0 for node in graph}

    # Random n_steps taken
    for _ in range(args.repeats):
         # Start point at a random node
         current_node = random.choice(list(graph.keys()))

         for _ in range(args.steps):
              
              # Increase visit count on the node walker lands in.
              visit_counts[current_node] += 1
              # Get outgoing links from current node
              neighbors= graph[current_node]

              # Restart from a random node if current node contains no outgoing links 
              if not neighbors:
                 current_node = random.choice(list(graph.keys()))

              else: current_node = random.choice(neighbors) 

    # Calculate probability
    total_visits = args.repeats * args.steps
    page_rank = {node: count / total_visits for node, count in visit_counts.items()}

    return page_rank



def distribution_page_rank(graph, args):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    args -- arguments named tuple

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """
    raise RuntimeError("This function is not implemented yet.")


parser = argparse.ArgumentParser(description="Estimates page ranks from link information")
parser.add_argument('datafile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help="Textfile of links among web pages as URL tuples")
parser.add_argument('-m', '--method', choices=('stochastic', 'distribution'), default='stochastic',
                    help="selected page rank algorithm")
parser.add_argument('-r', '--repeats', type=int, default=1_000_000, help="number of repetitions")
parser.add_argument('-s', '--steps', type=int, default=100, help="number of steps a walker takes")
parser.add_argument('-n', '--number', type=int, default=20, help="number of results shown")


if __name__ == '__main__':
    args = parser.parse_args()
    algorithm = distribution_page_rank if args.method == 'distribution' else stochastic_page_rank

    graph = load_graph(args)

    print_stats(graph)

    start = time.time()
    ranking = algorithm(graph, args)
   
    stop = time.time()
    time = stop - start


    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    sys.stderr.write(f"Top {args.number} pages:\n")
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:args.number]))
    sys.stderr.write(f"Calculation took {time:.2f} seconds.\n")

