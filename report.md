# 3D $N^2$-Queens Problem: MCMC Solution

## State Space Representation

We represent the state as an $N^2 \times 3$ matrix where each row contains the $(i, j, k)$ coordinates of one queen:
$$\mathbf{s} = \begin{bmatrix} i_1 & j_1 & k_1 \\ i_2 & j_2 & k_2 \\ \vdots & \vdots & \vdots \\ i_{N^2} & j_{N^2} & k_{N^2} \end{bmatrix}$$

where $i_m, j_m, k_m \in \{1, 2, \ldots, N\}$ for each queen $m$. This representation is chosen because it allows efficient position updates (change one row to move one queen) and fast conflict checking between any two queens (simple coordinate comparison).



## Attack Conditions

Two queens at positions $(i_1, j_1, k_1)$ and $(i_2, j_2, k_2)$ attack each other if any of the following conditions hold:

**Plane Attacks (exactly two coordinates match):**
- Same $i-j$ plane: $i_1 = i_2 \land j_1 = j_2 \land k_1 \neq k_2$
- Same $i-k$ plane: $i_1 = i_2 \land k_1 = k_2 \land j_1 \neq j_2$
- Same $j-k$ plane: $j_1 = j_2 \land k_1 = k_2 \land i_1 \neq i_2$

**2D Diagonals (within planes):**
- In $i-j$ plane: $|i_1 - i_2| = |j_1 - j_2| \land k_1 = k_2 \land |i_1 - i_2| \neq 0$
- In $i-k$ plane: $|i_1 - i_2| = |k_1 - k_2| \land j_1 = j_2 \land |i_1 - i_2| \neq 0$
- In $j-k$ plane: $|j_1 - j_2| = |k_1 - k_2| \land i_1 = i_2 \land |j_1 - j_2| \neq 0$

**3D Diagonals:**
- $|i_1 - i_2| = |j_1 - j_2| = |k_1 - k_2| \land |i_1 - i_2| \neq 0$



## Energy Function

The energy function counts the total number of attacking queen pairs:

$$E(\mathbf{s}) = \sum_{m=1}^{N^2} \sum_{n=m+1}^{N^2} \mathbf{1}[\text{queens } m \text{ and } n \text{ attack each other}]$$

This function directly measures constraint violations. A configuration with energy zero represents a valid solution. The energy landscape is smooth, meaning incremental moves tend to produce gradual energy changes rather than sudden jumps, which aids the MCMC algorithm in finding minima.


## Algorithm: Computing Energy

```
function computeEnergy(state):
    energy ← 0
    for m = 1 to N²:
        for n = m+1 to N²:
            if attacksEachOther(state[m], state[n]):
                energy ← energy + 1
    return energy

function attacksEachOther(pos₁, pos₂):
    (i₁, j₁, k₁) ← pos₁
    (i₂, j₂, k₂) ← pos₂
    
    // Plane attacks
    if (i₁ = i₂ ∧ j₁ = j₂ ∧ k₁ ≠ k₂): return TRUE
    if (i₁ = i₂ ∧ k₁ = k₂ ∧ j₁ ≠ j₂): return TRUE
    if (j₁ = j₂ ∧ k₁ = k₂ ∧ i₁ ≠ i₂): return TRUE
    
    // 2D diagonals
    if (|i₁ - i₂| = |j₁ - j₂| ∧ k₁ = k₂ ∧ |i₁ - i₂| ≠ 0): return TRUE
    if (|i₁ - i₂| = |k₁ - k₂| ∧ j₁ = j₂ ∧ |i₁ - i₂| ≠ 0): return TRUE
    if (|j₁ - j₂| = |k₁ - k₂| ∧ i₁ = i₂ ∧ |j₁ - j₂| ≠ 0): return TRUE
    
    // 3D diagonals
    if (|i₁ - i₂| = |j₁ - j₂| = |k₁ - k₂| ∧ |i₁ - i₂| ≠ 0): return TRUE
    
    return FALSE
```

For MCMC efficiency, when proposing to move queen $m$ to a new position, compute delta energy in $O(N²)$ instead of recomputing all $O(N⁴)$ pairs:

$$E_{\text{new}} - E_{\text{old}} = (\text{conflicts involving queen } m \text{ at new position}) - (\text{conflicts involving queen } m \text{ at old position})$$

This localizes computation to only the affected queen, making the algorithm practical for larger board sizes.



## MCMC Integration

### Target Distribution

Define the probability distribution over configurations:

$$\pi(s) = \frac{1}{Z} e^{-\beta E(s)}$$

where $Z = \sum_{s'} e^{-\beta E(s')}$ is the normalizing constant (partition function), $E(s)$ is the energy, and $\beta$ is the inverse temperature. Configurations with lower energy have exponentially higher probability.


### Metropolis-Hastings Algorithm

The algorithm samples from $\pi(s)$ by maintaining a chain of states $s_0, s_1, s_2, \ldots$ At iteration $t$:

**1. Propose:** Sample candidate $s'$ from proposal distribution $q(s' | s_t)$. In our case, $q$ selects a random queen uniformly and proposes a uniform random position.

**2. Accept/Reject:** Compute the acceptance probability:

$$\alpha(s_t \to s') = \min\left(1, \frac{\pi(s') q(s_t | s')}{\pi(s_t) q(s' | s_t)}\right)$$

Since our proposal is symmetric ($q(s_t | s') = q(s' | s_t)$), this simplifies to:

$$\alpha(s_t \to s') = \min\left(1, e^{-\beta(E(s') - E(s_t))}\right) = \min(1, e^{-\beta \Delta E})$$

Draw $u \sim \text{Uniform}(0,1)$. If $u < \alpha$, set $s_{t+1} = s'$. Otherwise, set $s_{t+1} = s_t$.


### Convergence Analysis

As $t \to \infty$, the chain converges to the stationary distribution $\pi(s)$. The convergence rate depends on $\beta$:

- Small $\beta$: High mixing, rapid convergence to high-temperature distribution (broad exploration)
- Large $\beta$: Slow mixing, concentrated probability on low-energy states (focused search)

For finite computation, simulated annealing increases $\beta$ over time:

$$\beta(t) = \beta_0 \cdot c^t$$

where $c > 1$ (typically $c = 1.001$). Early iterations explore broadly; later iterations concentrate on low-energy regions.



### Algorithm Summary

```
Initialize: s ← random configuration, β ← β₀
for t = 1 to T:
    m ← random queen index
    s' ← copy of s with queen m at random position
    ΔE ← computeDeltaEnergy(s, m, new position)
    if log(U[0,1]) < -β · ΔE:
        s ← s'
    β ← β · c
    record E(s)
```

The algorithm terminates when $E(s) = 0$ or computational budget exhausted.



### Key Properties

**Ergodicity:** The chain can reach any state from any other state, ensuring exploration of the entire state space.

**Aperiodicity:** The acceptance/rejection mechanism introduces randomness, preventing periodic cycling through states.

**Stationary Distribution:** In the limit, $P_\infty(s) = \pi(s) = e^{-\beta E(s)} / Z$.

Together, these guarantee that the chain asymptotically samples from $\pi(s)$, spending more time in low-energy regions as $\beta$ increases.