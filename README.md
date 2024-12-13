# PageRank Estimation Project

This project implements and optimizes algorithms for estimating PageRank values based on different graph representations. The two main algorithims implemented are:
1. **Stochastic PageRank**
2. **Distribution-based PageRank**

---

## **Features**
- **Graph Representations**: The project supports multiple graph representations, including:
  - Adjacency List
  - Weighted Adjacency List
  - Edge List
  - Adjacency Matrix
- **Algorithms**:
  - Stochastic PageRank: estimates the Page Rank by counting how frequently a random walk that starts on a random node will end up on each node on a given graph 
  - Distribution-based PageRank: estimates the Page Rank by iteratively calculating the probability that a random walker is currently on any node.
- **Optimizations**:
  - Integrated **NumPy** for faster numerical operations.
  - Implemented convergence checks to stop early when results stabilize.
  

---
## ** Command-line Arguments**

Following arguments can modify Pagerank calculations:

- `datafile`: Text file with links between web pages (URL tuples). Defaults to reading from `stdin`.
- `-m` or `--method`: Choose the PageRank algorithm (`stochastic` or `distribution`). Default: `stochastic`.
- `-r` or `--repeats`: Number of repetitions for the stochastic method. Default: `1,000,000`.
- `-s` or `--steps`: Number of steps for the distribution method. Default: `100`.
- `-n` or `--number`: Number of top results to display. Default: `20`.
- `-g` or `--representation`: Graph representation (`adjacency_list`, `edge_list`, `matrix`, `weighted_adjacency_list`). Default: `adjacency_list`.

## **Examples**

Here are several example commands to demonstrate the functionality of the algorithims with different graph representations:

```bash
# Stochastic PageRank using adjacency list
python3 page_rank.py school_web2024-1.txt -m stochastic -g adjacency_list -r 1000

# Distribution-based PageRank using adjacency list
python3 page_rank.py school_web2024-1.txt -m distribution -g adjacency_list -s 100

# Stochastic PageRank using edge list
python3 page_rank.py school_web2024-1.txt -m stochastic -g edge_list -r 1000

# Distribution-based PageRank using adjacency matrix
python3 page_rank.py school_web2024-1.txt -m distribution -g matrix -s 100

# Weighted adjacency list representation
python3 page_rank.py school_web2024-1.txt -g weighted_adjacency_list

## Acknowledgements

This project was created as part of the CSC1034 assessment. 

