# Solving the $3D$ $N^2$-Queens problem with MCMC

This repository contains a **Monte-Carlo Markov Chain** (MCMC) implementation for solving the $3D$ $N^2$-Queens problem using the **Metropolis-Hastings** algorithm with **simulated annealing**, alongside a web-based 3D visualization that displays the algorithm progression in real-time.

## Problem Description

The $3D$ $N^2$-Queens problem extends the classic $N$-Queens puzzle to $3$ dimensions. The goal is to place $N^2$ queens on an $N \times N \times N$ chessboard such that the least number of queens attack each other.

## Run Instructions

1. Create a python environment and install required packages.

    ```bash
    conda create -n queens python=3.10 && conda activate queens
    pip install -r requirements.txt
    ```

2. There are two different ways to launch the algorithm.

    - You can run the code as a **Python script**. In this case, again there are two different ways to proceed.

        - Run the code for a single configuration (board size, number of iterations, beta function, acceptance function). Add `--csv_output` to save the final queen positions to a CSV file.

            ```bash
            python src/mcmc.py --num_iterations <num-iterations> --board-size <board-size> --beta_func <beta-func> --acceptance_func <acceptance-func>
            ```

        - Get general results of our work comparing several configurations for given board sizes and number of iterations. *(Use `--help` to see all available options)*.

            ```bash
            python src/results.py --board-sizes <space-separated-board-sizes> --num-iterations <num-iterations>
            ```

    - You can launch the web-based visualization, as a **Flask application**. This displays queen moving positions in a 3D cube, with the energy evolution as the algorithm run. After running the command below, you should open your browser to `http://127.0.0.1:5000` and click "Start MCMC Simulation" to configure and run the algorithm.

        ```bash
        cd webapp
        python app.py
        ```

## Approach

This project uses MCMC with the following components:

1. **Metropolis-Hastings Algorithm**, which is a probabilistic acceptance criterion for proposed moves which accept moves that lower so-called energy, which is our loss function. It accepts worse moves probabilistically based on temperature, and acceptance probability is given by $P(\text{accept}) = \min(1, e^{-\beta \Delta E})$.

2. **Simulated Annealing**, which uses an exponential cooling schedule to control temperature ($\beta$ parameter) and helps escape local minima early in the search.

### Implementation details

#### State Space

We represent each state of our Markov Chain as a list of $N^2$ queens, where each queen is represented by its $(i, j, k)$ coordinates. This representation allows efficient position updates (change one row to move one queen).

#### Attack Conditions

Two queens at positions $(i_1, j_1, k_1)$ and $(i_2, j_2, k_2)$ attack each other if any of the following conditions hold.

##### Plane attacks

Two coordinates match.

- Same $i-j$ plane: $i_1 = i_2 \land j_1 = j_2 \land k_1 \neq k_2$
- Same $i-k$ plane: $i_1 = i_2 \land k_1 = k_2 \land j_1 \neq j_2$
- Same $j-k$ plane: $j_1 = j_2 \land k_1 = k_2 \land i_1 \neq i_2$

##### 2D diagonals

Two coordinates differ by the same amount, and the third coordinate matches.

- In $i-j$ plane: $|i_1 - i_2| = |j_1 - j_2| \land k_1 = k_2 \land |i_1 - i_2| \neq 0$
- In $i-k$ plane: $|i_1 - i_2| = |k_1 - k_2| \land j_1 = j_2 \land |i_1 - i_2| \neq 0$
- In $j-k$ plane: $|j_1 - j_2| = |k_1 - k_2| \land i_1 = i_2 \land |j_1 - j_2| \neq 0$

##### 3D diagonals

Three coordinates differ by the same amount.

- $|i_1 - i_2| = |j_1 - j_2| = |k_1 - k_2| \land |i_1 - i_2| \neq 0$

#### Energy Function

The energy function counts the total number of attacking queen pairs.

$$E(\mathbf{s}) = \sum_{m=1}^{N^2} \sum_{n=m+1}^{N^2} \mathbf{1}[\text{queens } m \text{ and } n \text{ attack each other}]$$

This function directly measures constraint violations. The energy landscape is smooth, meaning incremental moves tend to produce gradual energy changes rather than sudden jumps, which aids the MCMC algorithm in finding minima.

For MCMC efficiency, energy computation is optimized by using several hash maps to store the number of queens attacking each plane, face diagonal and space diagonal. This allows computation of initial energy in $O(N^2)$, which is the number of queens, instead of $O(N^4)$ pairs. Also, this allows computation of delta energy in $O(1)$ instead of $O(N^2)$ pairs, since a transition between two states involves moving a **single** queen to a new **unoccupied** position.

#### Metropolis-Hastings Algorithm

Define the probability distribution over configurations:

$$\pi(s) = \frac{1}{Z} e^{-\beta E(s)}$$

where $Z = \sum_{s'} e^{-\beta E(s')}$ is the normalizing constant (partition function), $E(s)$ is the energy, and $\beta$ is the inverse temperature. Configurations with lower energy have exponentially higher probability. The algorithm samples from $\pi(s)$ by maintaining a chain of states $s_0, s_1, s_2, \ldots$ At iteration $t$:

1. **Propose** Sample candidate $s'$ from proposal distribution $q(s' | s_t)$. In our case, $q$ selects a random queen uniformly and proposes a uniform random position.

2. **Accept/Reject** Compute the acceptance probability:

$$\alpha(s_t \to s') = \min\left(1, \frac{\pi(s') q(s_t | s')}{\pi(s_t) q(s' | s_t)}\right)$$

Since our proposal is symmetric ($q(s_t | s') = q(s' | s_t)$), this simplifies to:

$$\alpha(s_t \to s') = \min\left(1, e^{-\beta(E(s') - E(s_t))}\right) = \min(1, e^{-\beta \Delta E})$$

Draw $u \sim \text{Uniform}(0,1)$. If $u < \alpha$, set $s_{t+1} = s'$. Otherwise, set $s_{t+1} = s_t$.

#### Convergence Analysis

As $t \to \infty$, the chain converges to the stationary distribution $\pi(s)$. The convergence rate depends on $\beta$:

- Small $\beta$: High mixing, rapid convergence to high-temperature distribution (broad exploration)
- Large $\beta$: Slow mixing, concentrated probability on low-energy states (focused search)

For finite computation, simulated annealing increases $\beta$ over time:

$$\beta(t) = \beta_0 \cdot c^t$$

where $c > 1$ (typically $c = 1.001$). Early iterations explore broadly; later iterations concentrate on low-energy regions.