from dataclasses import dataclass
from utils import (
    compute_energy,
    compute_delta_energy,
    exponential_beta,
    metropolis_hastings,
)
from tqdm.notebook import tqdm
import numpy as np



@dataclass
class MCMCResult:
    """Results from MCMC run."""

    states: list
    energies: list
    betas: list
    accepted_moves: int
    final_state: list
    min_energy: float


def initialize_state(board_size):
    """Initialize with N² queens placed randomly (no two queens same cell)."""
    occupied = set()
    state = []
    num_queens = board_size**2
    for _ in range(num_queens):
        while True:
            pos = tuple(np.random.randint(1, board_size + 1) for _ in range(3))
            if pos not in occupied:
                occupied.add(pos)
                state.append(pos)
                break
    return state


def mcmc_chain(
    board_size,
    num_iterations,
    beta_func=exponential_beta,
    acceptance_func=metropolis_hastings,
    target_energy=0,
    verbose=True,
):
    """
    Run MCMC chain for the 3D N^2-Queens problem.

    Args:
        board_size: Size of board (N for N×N×N)
        num_iterations: Number of MCMC steps
        beta_func: Function that returns β given iteration number
        acceptance_func: Function that decides whether to accept a move
        target_energy: Stop when energy <= this value
        verbose: Print progress

    Returns:
        MCMCResult object containing chain history
    """
    state = initialize_state(board_size)
    energy = compute_energy(state)

    states = [state.copy()]
    energies = [energy]
    betas = []
    accepted_moves = 0

    pbar = tqdm(range(num_iterations), position=0, leave=True, desc="MCMC Chain")
    for t in pbar:
        beta = beta_func(t)
        betas.append(beta)

        # Move random queen to random position
        queen_idx = np.random.randint(0, len(state))
        new_pos = tuple(np.random.randint(1, board_size + 1) for _ in range(3))

        # Compute energy change
        delta_energy = compute_delta_energy(state, queen_idx, new_pos)

        # Accept or reject using Metropolis-Hastings
        if acceptance_func(delta_energy, beta):
            state[queen_idx] = new_pos
            energy += delta_energy
            accepted_moves += 1

        energies.append(energy)
        states.append(state.copy())

        # Stop if solution found
        if energy <= target_energy:
            if verbose:
                pbar.write(f"✓ Solution found at iteration {t} with energy {energy}")
            break

        if verbose and (t + 1) % 1000 == 0:
            acceptance_rate = 100 * accepted_moves / (t + 1)
            pbar.write(
                f"Iteration {t+1}: Energy={energy}, β={beta:.4f}, "
                f"Acceptance rate={acceptance_rate:.1f}%"
            )

    if verbose and energy > target_energy:
        pbar.write(f"✗ Stopped at iteration {num_iterations} with energy {energy}")

    return MCMCResult(
        states=states,
        energies=energies,
        betas=betas,
        accepted_moves=accepted_moves,
        final_state=state,
        min_energy=min(energies),
    )
