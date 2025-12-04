"""
Definition of utility functions for the 3D N^2-Queens problem.
"""

import matplotlib.pyplot as plt

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(energies, betas, accepted_moves, board_size):
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
        f"β Growth: {betas[-1]/betas[0]:.2f}x"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()