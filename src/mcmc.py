"""
Definition of the MCMC Metropolis-Hastings algorithm for the 3D N^2-Queens problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from queens import QueenState

# ============================================================================
# ACCEPTANCE CRITERION
# ============================================================================

def metropolis_hastings(delta_energy, beta):
    """Standard Metropolis-Hastings: always accept if ΔE < 0, else accept with prob exp(-β·ΔE)."""
    if delta_energy < 0:
        return True
    return np.random.random() < np.exp(-beta * delta_energy)


def always_accept(delta_energy, beta):
    """Always accept moves (for testing/debugging)."""
    return True

def greedy(delta_energy, beta):
    """Only accept improving moves."""
    return delta_energy < 0

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
# MAIN ALGORITHM
# ============================================================================

def mcmc_chain(board_size, num_iterations, target_energy=0, beta_func=exponential_beta, acceptance_func=metropolis_hastings, verbose=True):
    """Run MCMC chain for the 3D N^2-Queens problem."""
    
    state = QueenState(board_size=board_size)
    energy = state.initial_energy
    
    queens_positions = [state.queens.copy()]
    energies = [energy]
    betas = []
    accepted_moves = 0
    
    pbar = tqdm(range(num_iterations), desc="MCMC Iterations", leave=False, disable=not verbose)
    for it in pbar:
        beta = beta_func(it)
        betas.append(beta)
        
        # move random queen to random unoccupied position
        queen_idx, new_pos = state.new_queen_position()
        
        # compute energy change
        delta_energy = state.compute_delta_energy(queen_idx, new_pos)
        
        # accept or reject using Metropolis-Hastings
        if acceptance_func(delta_energy, beta):
            state.apply_move(queen_idx, new_pos)
            energy += delta_energy
            accepted_moves += 1
            
        queens_positions.append(state.queens.copy())
        energies.append(energy)
        
        if energy <= target_energy:
            if verbose:
                pbar.write(f"✓ Solution found at iteration {it} with energy {energy}")
            break
        
        if it + 1000 == 0:
            acceptance_rate = 100 * accepted_moves / (it + 1)
            if verbose:
                pbar.write(
                    f"Iteration {it + 1}: Energy={energy}, β={beta:.4f}, "
                    f"Acceptance rate={acceptance_rate:.1f}%"
                )
            
    if energy > target_energy and verbose:
        pbar.write(f"✗ Stopped at iteration {num_iterations} with energy {energy}")
        
    return queens_positions, energies, betas, accepted_moves

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_results(energies, betas, accepted_moves, board_size):
    """Create comprehensive visualization of a single MCMC run."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MCMC 3D N-Queens Solver (N={board_size})', fontsize=16, fontweight='bold')
    
    # Energy trajectory
    ax = axes[0, 0]
    ax.plot(energies, linewidth=1.5, color='steelblue', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Solution')
    ax.fill_between(range(len(energies)), energies, alpha=0.3, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Attacking Pairs)')
    ax.set_title('Energy Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature schedule
    ax = axes[0, 1]
    ax.semilogy(betas, linewidth=2, color='coral')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('β (Inverse Temperature)')
    ax.set_title('Temperature Schedule (log scale)')
    ax.grid(True, alpha=0.3, which='both')
    
    # Energy distribution
    ax = axes[1, 0]
    recent_energies = energies[-min(500, len(energies)):]
    ax.hist(recent_energies, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency')
    ax.set_title('Energy Distribution (Recent Iterations)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = (
        f"Final Energy: {energies[-1]}\n"
        f"Total Iterations: {len(energies)}\n"
        f"Accepted Moves: {accepted_moves}\n"
        f"Acceptance Rate: {100*accepted_moves/len(energies):.1f}%\n"
        f"Initial Energy: {energies[0]}\n"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MCMC N^2-Queens solver.")
    parser.add_argument("--num_iterations", type=int, required=True, help="Number of MCMC iterations.")
    parser.add_argument("--board_size", type=int, required=True, help="Size of one dimension of the 3D board.")
    parser.add_argument("--beta_func", type=str, default="exponential", help="Beta function to use.", choices=["exponential", "linear", "constant"])
    parser.add_argument("--acceptance_func", type=str, default="metropolis", help="Acceptance function to use.", choices=["metropolis", "always_accept", "greedy"])
    
    args = parser.parse_args()
    beta_func = exponential_beta if args.beta_func == "exponential" else linear_beta if args.beta_func == "linear" else constant_beta
    acceptance_func = metropolis_hastings if args.acceptance_func == "metropolis" else always_accept if args.acceptance_func == "always_accept" else greedy
    queens_positions, energies, betas, accepted_moves = mcmc_chain(
        board_size=args.board_size, 
        num_iterations=args.num_iterations, 
        beta_func=beta_func,
        acceptance_func=acceptance_func
    )
    plot_results(
        energies=energies,
        betas=betas,
        accepted_moves=accepted_moves,
        board_size=args.board_size
    )