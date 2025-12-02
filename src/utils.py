import numpy as np
from tqdm.notebook import tqdm

# ============================================================================
# ENERGY FUNCTION
# ============================================================================

def attacks_each_other(pos1, pos2):
    """Check if two queens attack each other in 3D."""
    i1, j1, k1 = pos1
    i2, j2, k2 = pos2

    # Same position
    if i1 == i2 and j1 == j2 and k1 == k2:
        return True    
    # Plane attacks (exactly two coordinates match)
    if i1 == i2 and j1 == j2 and k1 != k2:
        return True
    if i1 == i2 and k1 == k2 and j1 != j2:
        return True
    if j1 == j2 and k1 == k2 and i1 != i2:
        return True
    
    # 2D diagonals within planes
    if abs(i1 - i2) == abs(j1 - j2) and k1 == k2 and abs(i1 - i2) != 0:
        return True
    if abs(i1 - i2) == abs(k1 - k2) and j1 == j2 and abs(i1 - i2) != 0:
        return True
    if abs(j1 - j2) == abs(k1 - k2) and i1 == i2 and abs(j1 - j2) != 0:
        return True
    
    # 3D diagonals
    if (abs(i1 - i2) == abs(j1 - j2) == abs(k1 - k2) and abs(i1 - i2) != 0):
        return True
    
    return False


def compute_energy(state, verbose=False):
    """Compute total number of attacking pairs."""
    conflicts = 0
    pbar = tqdm(range(len(state)), disable=not verbose, desc="Computing initial energy")

    for m in pbar:
        for n in range(m + 1, len(state)):
            if attacks_each_other(state[m], state[n]):
                conflicts += 1
    return conflicts


def compute_delta_energy(state, queen_idx, new_pos):
    """Compute energy change when moving one queen. O(N²) instead of O(N⁴)."""
    old_pos = state[queen_idx]
    conflicts_before = sum(1 for i in range(len(state)) 
                          if i != queen_idx and attacks_each_other(old_pos, state[i]))
    conflicts_after = sum(1 for i in range(len(state)) 
                         if i != queen_idx and attacks_each_other(new_pos, state[i]))
    return conflicts_after - conflicts_before


# ============================================================================
# TEMPERATURE SCHEDULES
# ============================================================================

def exponential_beta(iteration, beta_0=0.1, cooling_rate=1.001):
    """β(t) = β₀ · c^t"""
    return beta_0 * (cooling_rate ** iteration)


def linear_beta(iteration, beta_0=0.1, t_max=10000):
    """β(t) = β₀ · (1 + t/t_max)"""
    return beta_0 * (1 + iteration / t_max)

def constant_beta(iteration, beta_0=0.1):
    """β(t) = β₀"""
    return beta_0


# ============================================================================
# ACCEPTANCE CRITERION
# ============================================================================

def metropolis_hastings(delta_energy, beta):
    """Standard Metropolis-Hastings: always accept if ΔE < 0, else accept with prob exp(-β·ΔE)."""
    if delta_energy < 0:
        return True
    return np.random.random() < np.exp(-beta * delta_energy)


