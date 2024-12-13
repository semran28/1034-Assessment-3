# Optimization Report

This document outlines several optimization strategies implemented to increase the execution speed and efficiency of my PageRank calculations. Below are the methods applied:

## 1. NumPy with Distribution-Based PageRank
- The adjacency matrix was converted into a NumPy array for faster numerical operations.
- **Performance Improvement**: Execution speed improved from **1.29 seconds** to **0.01 seconds** for the distribution-based PageRank algorithm.

## 2. Convergence Check
- Implemented a convergence check to stop the PageRank algorithm when the rank values began to stabilize (converge). 
- This reduced unnecessary computations and improved efficiency.
- **Impact**: Overall execution time was significantly reduced with this method.

## 3. NumPy with Stochastic Algorithm
- Replaced Python’s `random.choice` with NumPy’s `np.random.choice` in the `stochastic_page_rank_edge_list` function.
- Using `random.choice` in each line was proving inefficient and slow.
- **Performance Improvement**: Execution time reduced from **0.24 seconds** to **0.13 seconds**.

## Conclusion
With these implemented strategies, I significantly improved the performance of my PageRank calculations. NumPy proved to be the most impactful tool in optimizing and simplifying my code effectively. This task highlighted the importance of thorough testing and optimization, particularly when working with algorithms designed to handle large data files.
