"""
Definition of the MCMC Metropolis-Hastings algorithm for the 3D N^2-Queens problem.
"""

import numpy as np
from tqdm import tqdm
from queens import QueenState
from utils import visualize_results

# ============================================================================
# ACCEPTANCE CRITERION
# ============================================================================

def _metropolis_hastings(delta_energy, beta):
    """Standard Metropolis-Hastings: always accept if ΔE < 0, else accept with prob exp(-β·ΔE)."""
    
    if delta_energy < 0:
        return True
    return np.random.random() < np.exp(-beta * delta_energy)


# ============================================================================
# TEMPERATURE SCHEDULES
# ============================================================================

def exponential_beta(iteration, beta_0=0.1, cooling_rate=1.001):
    """β(t) = β₀ · c^t"""
    return beta_0 * (cooling_rate ** iteration)


def linear_beta(iteration, beta_0=0.1, t_max=10000):
    """β(t) = β₀ · (1 + t/t_max)"""
    return beta_0 * (1 + iteration / t_max)


# ============================================================================
# MAIN ALGORITHM
# ============================================================================

def mcmc_chain(board_size, num_iterations, target_energy=0, beta_func=exponential_beta):
    """Run MCMC chain for the 3D N^2-Queens problem."""
    
    import time
    start_time = time.time()
    state = QueenState(board_size)
    end_time = time.time()
    print(f"Time taken to initialize state: {end_time - start_time} seconds")

    energy = state.initial_energy
    
    queens_positions = [state.queens.copy()]
    energies = [energy]
    betas = []
    accepted_moves = 0
    
    pbar = tqdm(range(num_iterations), desc="MCMC Iterations", leave=True)
    for it in pbar:
        beta = beta_func(it)
        betas.append(beta)
        
        # move random queen to random unoccupied position
        queen_idx, new_pos = state.new_queen_position()
        
        # compute energy change
        delta_energy = state.compute_delta_energy(queen_idx, new_pos)
        
        # accept or reject using Metropolis-Hastings
        if _metropolis_hastings(delta_energy, beta):
            state.apply_move(queen_idx, new_pos)
            energy += delta_energy
            accepted_moves += 1
            
        queens_positions.append(state.queens.copy())
        energies.append(energy)
        
        if energy <= target_energy:
            pbar.write(f"✓ Solution found at iteration {it} with energy {energy}")
            break
        
        if it + 1000 == 0:
            acceptance_rate = 100 * accepted_moves / (it + 1)
            pbar.write(
                f"Iteration {it + 1}: Energy={energy}, β={beta:.4f}, "
                f"Acceptance rate={acceptance_rate:.1f}%"
            )
            
    if energy > target_energy:
        pbar.write(f"✗ Stopped at iteration {num_iterations} with energy {energy}")
        
    return queens_positions, energies, betas, accepted_moves


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MCMC N^2-Queens solver.")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of MCMC iterations.")
    parser.add_argument("--board_size", type=int, required=True, help="Size of one dimension of the 3D board.")
    parser.add_argument("--beta_func", type=str, default="exponential", help="Beta function to use.")
    
    args = parser.parse_args()
    beta_func = exponential_beta if args.beta_func == "exponential" else linear_beta
    queens_positions, energies, betas, accepted_moves = mcmc_chain(
        board_size=args.board_size, 
        num_iterations=args.num_iterations, 
        beta_func=beta_func
    )
    visualize_results(
        energies=energies,
        betas=betas,
        accepted_moves=accepted_moves,
        board_size=args.board_size
    )