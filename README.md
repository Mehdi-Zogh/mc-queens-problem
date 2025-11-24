# 3D $N^2$-Queens MCMC Solver

A Markov Chain Monte Carlo (MCMC) implementation for solving the 3D N-Queens problem using simulated annealing and the Metropolis-Hastings algorithm.

## Problem Description

The 3D N-Queens problem extends the classic N-Queens puzzle to three dimensions. The goal is to place N² queens on an N×N×N chessboard such that no two queens attack each other. In 3D space, two queens attack each other if they share:
- 2 commmon coordinates
- a 2D diagonal 
- a 3D diagonal

## Solution Approach

This project uses **Markov Chain Monte Carlo (MCMC)** with the following components:

### 1. **Simulated Annealing**
- Uses an exponential cooling schedule to control temperature (β parameter)
- Temperature decreases over iterations, reducing exploration and focusing on exploitation
- Helps escape local minima early in the search

### 2. **Metropolis-Hastings Algorithm**
- Probabilistic acceptance criterion for proposed moves
- Accept moves that lower energy (conflicts)
- Accept worse moves probabilistically based on temperature
- Acceptance probability: $P(\text{accept}) = \min(1, e^{-\beta \Delta E})$

### 3. **Energy Function**
- Energy = number of queen conflicts
- Goal: minimize energy to 0 (no conflicts)
- Efficiently computed using delta updates for each move

## Project Structure

```
project/
├── README.md             
├── report.md             
├── src/
│   ├── __init__.py       
│   ├── mcmc.py           
│   ├── utils.py          
│   ├── visualization.py  
│   └── experiments.ipynb 
```