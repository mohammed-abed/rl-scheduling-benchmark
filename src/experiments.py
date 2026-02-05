
"""
Experimental Framework for Comparing Exploration Strategies
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import pickle
from pathlib import Path
from tqdm import tqdm

from .environment import ResourceSchedulingEnv
from .agent import QLearningAgent


def run_single_trial(
    exploration_strategy: str,
    n_episodes: int = 500,
    random_seed: int = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run a single trial with a specific exploration strategy.
    
    Args:
        exploration_strategy: Exploration strategy to use
        n_episodes: Number of episodes to run
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (episode_rewards, completion_time_data)
    """
    env = ResourceSchedulingEnv(random_seed=random_seed)
    agent = QLearningAgent(
        n_actions=env.n_servers,
        exploration_strategy=exploration_strategy,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        random_seed=random_seed
    )
    
    episode_rewards = []
    completion_times = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Decay epsilon for epsilon-greedy
        if exploration_strategy == 'epsilon-greedy':
            agent.decay_epsilon(episode, n_episodes)
        
        while not done:
            action = agent.select_action(state, episode)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        
        # Record completion times periodically
        if episode % 50 == 0:
            stats = env.get_statistics()
            if stats:
                completion_times.append({
                    'episode': episode,
                    'mean_completion_time': stats.get('mean_completion_time', 0),
                    'p95_completion_time': stats.get('p95_completion_time', 0)
                })
    
    return np.array(episode_rewards), completion_times


def run_experiment(
    exploration_strategy: str,
    n_episodes: int = 500,
    n_runs: int = 10,
    base_seed: int = 42
) -> Dict:
    """
    Run multiple trials and collect statistics.
    
    Args:
        exploration_strategy: Exploration strategy to use
        n_episodes: Number of episodes per run
        n_runs: Number of independent runs
        base_seed: Base random seed
        
    Returns:
        Dictionary with results
    """
    print(f"\nRunning {n_runs} trials with {exploration_strategy}...")
    
    all_rewards = []
    all_completion_times = []
    
    for run in tqdm(range(n_runs), desc=f"{exploration_strategy}"):
        seed = base_seed + run if base_seed is not None else None
        rewards, completion_times = run_single_trial(
            exploration_strategy=exploration_strategy,
            n_episodes=n_episodes,
            random_seed=seed
        )
        
        all_rewards.append(rewards)
        all_completion_times.extend([
            {**ct, 'run': run, 'strategy': exploration_strategy}
            for ct in completion_times
        ])
    
    all_rewards = np.array(all_rewards)
    
    # Calculate metrics
    final_performance = np.mean(all_rewards[:, -50:])  # Last 50 episodes
    final_std = np.std(all_rewards[:, -50:])
    
    # Find convergence point (when reach 95% of final performance)
    mean_rewards = np.mean(all_rewards, axis=0)
    convergence_threshold = final_performance * 0.95
    convergence_episode = np.argmax(mean_rewards >= convergence_threshold)
    
    # Area under curve (sample efficiency)
    auc = np.trapz(mean_rewards)
    
    return {
        'strategy': exploration_strategy,
        'rewards': all_rewards,
        'completion_times': all_completion_times,
        'final_performance_mean': final_performance,
        'final_performance_std': final_std,
        'convergence_episode': convergence_episode,
        'auc': auc,
        'mean_curve': mean_rewards,
        'std_curve': np.std(all_rewards, axis=0)
    }


def compare_strategies(
    strategies: List[str] = None,
    n_episodes: int = 500,
    n_runs: int = 10,
    base_seed: int = 42,
    output_dir: str = 'results'
) -> Dict:
    """
    Compare multiple exploration strategies.
    
    Args:
        strategies: List of strategies to compare
        n_episodes: Number of episodes per run
        n_runs: Number of independent runs
        base_seed: Base random seed
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all results
    """
    if strategies is None:
        strategies = ['epsilon-greedy', 'ucb', 'boltzmann']
    
    print("="*60)
    print("REINFORCEMENT LEARNING EXPLORATION STRATEGY COMPARISON")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Episodes per run: {n_episodes}")
    print(f"  Independent runs: {n_runs}")
    print(f"  Random seed: {base_seed}")
    
    results = {}
    
    # Run experiments for each strategy
    for strategy in strategies:
        results[strategy] = run_experiment(
            exploration_strategy=strategy,
            n_episodes=n_episodes,
            n_runs=n_runs,
            base_seed=base_seed
        )
    
    # Statistical analysis
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Print descriptive statistics
    print("\nFinal Performance (last 50 episodes):")
    print(f"{'Strategy':<15} {'Mean Reward':<15} {'Std Dev':<15} {'Convergence Ep':<15}")
    print("-" * 60)
    
    for strategy in strategies:
        data = results[strategy]
        print(f"{strategy:<15} {data['final_performance_mean']:<15.2f} "
              f"{data['final_performance_std']:<15.2f} {data['convergence_episode']:<15}")
    
    # Pairwise comparisons
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE (Independent t-tests)")
    print("="*60)
    
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            perf1 = results[s1]['rewards'][:, -50:].mean(axis=1)
            perf2 = results[s2]['rewards'][:, -50:].mean(axis=1)
            
            t_stat, p_value = stats.ttest_ind(perf1, perf2)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(perf1) + np.var(perf2)) / 2)
            cohens_d = (np.mean(perf1) - np.mean(perf2)) / pooled_std
            
            print(f"\n{s1} vs {s2}:")
            print(f"  Mean difference: {np.mean(perf1) - np.mean(perf2):.2f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_path / 'experiment_results.pkl'}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RL scheduling experiments')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Single strategy to test (default: all)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of episodes per run')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    strategies = [args.strategy] if args.strategy else ['epsilon-greedy', 'ucb', 'boltzmann']
    
    results = compare_strategies(
        strategies=strategies,
        n_episodes=args.episodes,
        n_runs=args.runs,
        base_seed=args.seed,
        output_dir=args.output
    )
