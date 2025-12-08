#!/usr/bin/env python3
import sys, os, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Callable

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from mcmc import mcmc_chain, exponential_beta, linear_beta, constant_beta, _metropolis_hastings, _greedy

# Plot style
for style in ['seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'default']:
    try: plt.style.use(style); break
    except: pass
sns.set_palette("husl")
plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 300, 'font.size': 11})

# Function mappings
BETA_FUNCS = {'exponential': exponential_beta, 'linear': linear_beta, 'constant': constant_beta}
ACCEPT_FUNCS = {'metropolis': _metropolis_hastings, 'greedy': _greedy}


@dataclass
class Result:
    board_size: int
    beta: str
    accept: str
    energies: List[List[float]]
    min_energies: List[float]


class Tester:
    def __init__(self, board_sizes, num_iterations, num_runs):
        self.board_sizes, self.num_iterations, self.num_runs = board_sizes, num_iterations, num_runs
        self.results = []

    def run(self):
        total = len(self.board_sizes) * len(BETA_FUNCS) * len(ACCEPT_FUNCS)
        with tqdm(total=total, desc="Running experiments") as pbar:
            for bs in self.board_sizes:
                for beta_name in BETA_FUNCS:
                    for accept_name in ACCEPT_FUNCS:
                        energies, mins = [], []
                        for _ in range(self.num_runs):
                            _, e, _, _ = mcmc_chain(bs, self.num_iterations, 0, BETA_FUNCS[beta_name], ACCEPT_FUNCS[accept_name])
                            energies.append(e)
                            mins.append(min(e))
                        self.results.append(Result(bs, beta_name, accept_name, energies, mins))
                        pbar.set_postfix({'N': bs, 'β': beta_name[:3], 'α': accept_name[:4]})
                        pbar.update(1)
        return self.results

    def _plot_comparison(self, filter_fn, x_fn, title, colors, save_path=None):
        results = [r for r in self.results if filter_fn(r)]
        if not results: return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        for i, r in enumerate(results):
            avg, std = np.mean(r.energies, axis=0), np.std(r.energies, axis=0)
            ax1.plot(avg, label=x_fn(r), linewidth=2, color=colors[i], alpha=0.9)
            ax1.fill_between(range(len(avg)), avg - std, avg + std, alpha=0.2, color=colors[i])
        ax1.set(xlabel='Iteration', ylabel='Energy', title='Energy Convergence', ylim=(0, None))
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        means = [np.mean(r.min_energies) for r in results]
        stds = [np.std(r.min_energies) for r in results]
        ax2.bar(range(len(results)), means, yerr=stds, color=colors[:len(results)], alpha=0.8, edgecolor='black', capsize=5)
        ax2.set_xticks(range(len(results)))
        ax2.set_xticklabels([x_fn(r) for r in results])
        ax2.set(ylabel='Min Energy', title='Best Energy Achieved')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig

    def plot_beta(self, bs, save_path=None):
        colors = sns.color_palette("husl", 3)
        return self._plot_comparison(
            lambda r: r.board_size == bs and r.accept == 'metropolis',
            lambda r: r.beta.capitalize(),
            f'Beta Comparison (N={bs}, Acceptance=Metropolis)', colors, save_path)

    def plot_accept(self, bs, save_path=None):
        colors = sns.color_palette("Set2", 2)
        return self._plot_comparison(
            lambda r: r.board_size == bs and r.beta == 'exponential',
            lambda r: r.accept.title(),
            f'Acceptance Comparison (N={bs}, Beta=Exponential)', colors, save_path)

    def plot_board(self, save_path=None):
        colors = sns.color_palette("viridis", len(self.board_sizes))
        return self._plot_comparison(
            lambda r: r.beta == 'exponential' and r.accept == 'metropolis',
            lambda r: f'N={r.board_size}',
            'Board Size Comparison (Beta=Exponential, Acceptance=Metropolis)', colors, save_path)

    def summary(self):
        return pd.DataFrame([{
            'Board Size': r.board_size, 'Beta': r.beta.capitalize(),
            'Acceptance': r.accept.title(), 'Min Energy': f"{np.mean(r.min_energies):.2f}"
        } for r in self.results])

    def save_plots(self, output_dir="results/plots"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to {output_dir}/...")
        for bs in self.board_sizes:
            self.plot_beta(bs, f'{output_dir}/beta_N{bs}.png'); plt.close()
            self.plot_accept(bs, f'{output_dir}/accept_N{bs}.png'); plt.close()
        if len(self.board_sizes) > 1:
            self.plot_board(f'{output_dir}/board_comparison.png'); plt.close()
        print("✓ All plots saved!")


def main():
    parser = argparse.ArgumentParser(description='MCMC N-Queens Testing')
    parser.add_argument('-b', '--board-sizes', type=int, nargs='+', default=[2], help='Board sizes')
    parser.add_argument('-i', '--iterations', type=int, default=1000, help='Iterations per run')
    parser.add_argument('-r', '--runs', type=int, default=2, help='Runs per config')
    parser.add_argument('-o', '--output-dir', type=str, default='results/plots', help='Output directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip plots')
    parser.add_argument('--show-plots', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    print(f"Board sizes: {args.board_sizes} | Iterations: {args.iterations:,} | Runs: {args.runs}")

    tester = Tester(args.board_sizes, args.iterations, args.runs)
    tester.run()
    print(tester.summary().to_string(index=False))

    if not args.no_plots:
        tester.save_plots(args.output_dir)
        if args.show_plots:
            for bs in args.board_sizes:
                tester.plot_beta(bs); tester.plot_accept(bs); plt.show()
            if len(args.board_sizes) > 1: tester.plot_board(); plt.show()

if __name__ == "__main__":
    main()
