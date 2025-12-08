"""
Final results generation and plotting for the 3D N^2-Queens problem.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from mcmc import mcmc_chain, exponential_beta, linear_beta, constant_beta, metropolis_hastings, greedy

# ============================================================================
# PLOTTING SETUP
# ============================================================================

import seaborn as sns
sns.set_palette("husl")

import matplotlib.pyplot as plt
for style in ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'default']:
    try: 
        plt.style.use(style)
        break
    except: 
        pass
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 11})

BETA_FUNCS = {'exponential': exponential_beta, 'linear': linear_beta, 'constant': constant_beta}
ACCEPT_FUNCS = {'metropolis': metropolis_hastings, 'greedy': greedy}

# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class Result:
    board_size: int
    beta: str
    accept: str
    energies: list[list[float]]
    min_energies: list[float]
    
# ============================================================================
# RESULTS GENERATION
# ============================================================================

def get_results(board_sizes, num_iterations, num_runs):
    """
    Run MCMC experiments for the given board sizes, number of iterations, and number of runs, with all possible beta and acceptance functions.
    """
    results = []
    total = len(board_sizes) * len(BETA_FUNCS) * len(ACCEPT_FUNCS)
    with tqdm(total=total, desc="MCMC Experiments") as pbar:
        for bs in board_sizes:
            for beta_name in BETA_FUNCS:
                for accept_name in ACCEPT_FUNCS:
                    energies, min_energies = [], []
                    for _ in range(num_runs):
                        _, e, _, _ = mcmc_chain(board_size=bs, num_iterations=num_iterations, target_energy=0, beta_func=BETA_FUNCS[beta_name], acceptance_func=ACCEPT_FUNCS[accept_name], verbose=False)
                        energies.append(e)
                        min_energies.append(min(e))
                    results.append(Result(board_size=bs, beta=beta_name, accept=accept_name, energies=energies, min_energies=min_energies))
                    pbar.update(1)
    return results

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_beta(results,board_size, save_path=None):
    """
    Plot energy as a function of the beta function for the given board size.
    """
    colors = sns.color_palette("husl", 3)
    return _plot_comparison(
        results=results,
        filter_fn=lambda r: r.board_size == board_size and r.accept == 'metropolis',
        x_fn=lambda r: r.beta.capitalize(),
        title=f'Beta Comparison (N={board_size}, Acceptance=Metropolis)', 
        colors=colors,
        save_path=save_path
    )
    
    
def plot_accept(results, board_size, save_path=None):
    """
    Plot energy as a function of the acceptance function for the given board size.
    """
    colors = sns.color_palette("Set2", 2)
    return _plot_comparison(
        results=results,
        filter_fn=lambda r: r.board_size == board_size and r.beta == 'exponential',
        x_fn=lambda r: r.accept.title(),
        title=f'Acceptance Comparison (N={board_size}, Beta=Exponential)', 
        colors=colors,
        save_path=save_path
    )


def plot_board(results, board_sizes, save_path=None):
    """
    Plot energy as a function of the board size.
    """
    colors = sns.color_palette("viridis", len(board_sizes))
    return _plot_comparison(
        results=results,
        filter_fn=lambda r: r.beta == 'exponential' and r.accept == 'metropolis',
        x_fn=lambda r: f'N={r.board_size}',
        title=f'Board Size Comparison (Beta=Exponential, Acceptance=Metropolis)', 
        colors= colors,
        save_path=save_path
    )


def save_plots(results, board_sizes, output_dir="results/plots"):
    """
    Save plots to the given output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to {output_dir}/...")
    for bs in board_sizes:
        plot_beta(results=results,board_size=bs, save_path=f'{output_dir}/beta_N{bs}.png')
        plt.close()
        plot_accept(results=results, board_size=bs, save_path=f'{output_dir}/accept_N{bs}.png')
        plt.close()
    if len(board_sizes) > 1:
        plot_board(results=results, board_sizes=board_sizes, save_path=f'{output_dir}/board_comparison.png')
        plt.close()
    print("✓ All plots saved!")


def _plot_comparison(results, filter_fn, x_fn, title, colors, save_path=None):
    """
    Plot a comparison of energies using given filter function, x function, title, colors, and save path.
    """
    results = [r for r in results if filter_fn(r)]
    if not results: 
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    for i, r in enumerate(results):
        avg, std = np.mean(r.energies, axis=0), np.std(r.energies, axis=0)
        ax1.plot(avg, label=x_fn(r), linewidth=2, color=colors[i], alpha=0.9)
        ax1.fill_between(range(len(avg)), avg - std, avg + std, alpha=0.2, color=colors[i])
    ax1.set(xlabel='Iteration', ylabel='Energy', title='Energy Convergence', ylim=(0, None))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    means = [np.mean(r.min_energies) for r in results]
    stds = [np.std(r.min_energies) for r in results]
    ax2.bar(range(len(results)), means, yerr=stds, color=colors[:len(results)], alpha=0.8, edgecolor='black', capsize=5)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([x_fn(r) for r in results])
    ax2.set(ylabel='Min Energy', title='Best Energy Achieved')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MCMC N^2-Queens testing.')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of MCMC iterations per run.')
    parser.add_argument('--board_sizes', type=int, nargs='+', default=[2], help='Sizes of one dimension of each 3D board.')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs per config.')
    parser.add_argument('--show_plots', action='store_true', help='Show plots interactively.')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to the results/ directory.')
    args = parser.parse_args()

    print(f"Board sizes: {args.board_sizes} | Nb. iterations: {args.num_iterations:,} | Nb. runs: {args.runs}")
    
    results = get_results(board_sizes=args.board_sizes, num_iterations=args.num_iterations, num_runs=args.runs)
    summary = pd.DataFrame([{
        'Board Size': r.board_size, 'Beta': r.beta.capitalize(),
        'Acceptance': r.accept.title(), 'Min Energy': f"{np.mean(r.min_energies):.2f}"
    } for r in results])
    print(summary.to_string(index=False))
    
    if args.save_plots:
        save_plots(results=results, board_sizes=args.board_sizes, output_dir="results")
    if args.show_plots:
        for bs in args.board_sizes:
            plot_beta(results=results, board_size=bs)
            plot_accept(results=results, board_size=bs)
            plt.show()
        if len(args.board_sizes) > 1:
            plot_board(results=results, board_sizes=args.board_sizes)
            plt.show()