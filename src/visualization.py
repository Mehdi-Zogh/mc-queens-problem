import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def visualize_results(result, board_size):
    """Create comprehensive visualization of MCMC run."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MCMC 3D N-Queens Solver (N={board_size})', fontsize=16, fontweight='bold')
    
    # Energy trajectory
    ax = axes[0, 0]
    ax.plot(result.energies, linewidth=1.5, color='steelblue', alpha=0.8)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Solution')
    ax.fill_between(range(len(result.energies)), result.energies, alpha=0.3, color='steelblue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (Attacking Pairs)')
    ax.set_title('Energy Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature schedule
    ax = axes[0, 1]
    ax.semilogy(result.betas, linewidth=2, color='coral')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('β (Inverse Temperature)')
    ax.set_title('Temperature Schedule (log scale)')
    ax.grid(True, alpha=0.3, which='both')
    
    # Energy distribution
    ax = axes[1, 0]
    recent_energies = result.energies[-min(500, len(result.energies)):]
    ax.hist(recent_energies, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Energy')
    ax.set_ylabel('Frequency')
    ax.set_title('Energy Distribution (Recent Iterations)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = (
        f"Final Energy: {result.min_energy}\n"
        f"Total Iterations: {len(result.energies)}\n"
        f"Accepted Moves: {result.accepted_moves}\n"
        f"Acceptance Rate: {100*result.accepted_moves/len(result.energies):.1f}%\n"
        f"Initial Energy: {result.energies[0]}\n"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig