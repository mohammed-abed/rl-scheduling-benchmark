
"""
Visualization utilities for experimental results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_learning_curves(results: Dict, save_path: str = None):
    """Plot learning curves for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'epsilon-greedy': '#1f77b4', 'ucb': '#ff7f0e', 'boltzmann': '#2ca02c'}
    
    for strategy, data in results.items():
        mean_rewards = data['mean_curve']
        std_rewards = data['std_curve']
        episodes = np.arange(len(mean_rewards))
        
        color = colors.get(strategy, None)
        ax.plot(episodes, mean_rewards, label=strategy, linewidth=2, color=color)
        ax.fill_between(episodes,
                        mean_rewards - std_rewards,
                        mean_rewards + std_rewards,
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Learning Curves by Exploration Strategy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_comparison(results: Dict, save_path: str = None):
    """Plot convergence speed comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    strategies = list(results.keys())
    convergence_episodes = [results[s]['convergence_episode'] for s in strategies]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(strategies)]
    bars = ax.bar(strategies, convergence_episodes, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Episodes to Convergence', fontsize=12)
    ax.set_title('Convergence Speed Comparison\n(Episodes to reach 95% of final performance)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved convergence comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_performance_distribution(results: Dict, save_path: str = None):
    """Plot final performance distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = list(results.keys())
    final_performances = [
        results[s]['rewards'][:, -50:].flatten() for s in strategies
    ]
    
    positions = range(len(strategies))
    
    # Violin plot
    parts = ax.violinplot(final_performances, positions=positions,
                          showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(strategies)]
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(strategies)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Final Performance Distribution\n(Last 50 episodes)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved performance distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_completion_times(results: Dict, save_path: str = None):
    """Plot job completion time evolution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combine all completion time data
    all_data = []
    for strategy, data in results.items():
        all_data.extend(data['completion_times'])
    
    df = pd.DataFrame(all_data)
    
    colors = {'epsilon-greedy': '#1f77b4', 'ucb': '#ff7f0e', 'boltzmann': '#2ca02c'}
    
    for strategy in results.keys():
        strategy_data = df[df['strategy'] == strategy]
        grouped = strategy_data.groupby('episode')['mean_completion_time'].mean()
        
        ax.plot(grouped.index, grouped.values, marker='o', markersize=4,
               label=strategy, linewidth=2, color=colors.get(strategy, None))
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Job Completion Time', fontsize=12)
    ax.set_title('Job Completion Time Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved completion times to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_figures(results: Dict, output_dir: str = 'results/figures'):
    """Generate all visualization figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating figures...")
    
    plot_learning_curves(results, output_path / 'learning_curves.png')
    plot_convergence_comparison(results, output_path / 'convergence_comparison.png')
    plot_performance_distribution(results, output_path / 'performance_distribution.png')
    plot_completion_times(results, output_path / 'completion_times.png')
    
    print(f"\nAll figures saved to {output_path}/")
