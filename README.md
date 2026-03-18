# Reinforcement Learning for Dynamic Task Scheduling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project investigates how different exploration strategies in Q-learning affect convergence speed and solution quality in dynamic multi-resource scheduling problems.

**Research Question:** How do ε-greedy, UCB, and Boltzmann exploration strategies compare in terms of convergence speed, final performance, and sample efficiency for resource scheduling tasks?

## Key Features

- **Realistic Environment**: Multi-server scheduling simulation with heterogeneous resources, dynamic job arrivals, and priority queuing
- **Multiple Algorithms**: Implementations of Q-learning with ε-greedy, Upper Confidence Bound (UCB), and Boltzmann (softmax) exploration
- **Rigorous Evaluation**: 10 independent runs per condition, statistical significance testing, effect size analysis
- **Comprehensive Metrics**: Convergence speed, final performance, job completion times, queue lengths

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/mohammed-abed/rl-scheduling-benchmark.git
cd rl-scheduling-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Run Full Experiment Suite

```bash
# Run all experiments (takes ~30 minutes)
python -m src.experiments --episodes 500 --runs 10

# Quick test run
python -m src.experiments --episodes 100 --runs 3
```

### Run Specific Strategy

```bash
# Test epsilon-greedy only
python -m src.experiments --strategy epsilon-greedy --episodes 500 --runs 10

# Test UCB
python -m src.experiments --strategy ucb --episodes 500 --runs 10

# Test Boltzmann
python -m src.experiments --strategy boltzmann --episodes 500 --runs 10
```

### Generate Visualizations

```bash
python scripts/generate_figures.py --results results/experiment_results.pkl
```

## Project Structure

```
src/
├── environment.py      # ResourceSchedulingEnv implementation
├── agent.py           # QLearningAgent with multiple exploration strategies
├── experiments.py     # Experimental framework and statistical analysis
└── visualization.py   # Plotting and figure generation

tests/
├── test_environment.py  # Unit tests for environment
└── test_agent.py       # Unit tests for agent

notebooks/
└── exploratory_analysis.ipynb  # Interactive analysis and exploration
```

## Results

### Main Findings


The environment runs correctly with all three strategies implemented. Preliminary single runs across 500 episodes suggest UCB converges faster than ε-greedy while Boltzmann achieves better asymptotic performance, consistent with theoretical predictions. Full statistical analysis across 10 replications per condition is pending.

### Visualizations

Results are automatically saved in `results/figures/`:
- `learning_curves.png` - Reward over episodes by strategy
- `convergence_comparison.png` - Bar chart of convergence speeds
- `performance_distribution.png` - Violin plots of final performance
- `completion_times.png` - Job completion time evolution

## Methodology

### Environment Details

- **State Space**: ~1,000 discrete states (queue length × server loads × job priority)
- **Action Space**: 5 actions (server assignment decisions)
- **Reward Function**: Negative job completion time weighted by priority
- **Dynamics**: Poisson job arrivals (λ=0.3), heterogeneous server capacities

### Experimental Design

- **Conditions**: 3 (ε-greedy, UCB, Boltzmann)
- **Replications**: 10 independent runs per condition
- **Episodes**: 500 per run
- **Evaluation**: Paired t-tests, Cohen's d effect sizes
- **Metrics**: 
  - Convergence speed (episode to 95% of final performance)
  - Final performance (mean reward over last 50 episodes)
  - Sample efficiency (area under learning curve)

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate (α) | 0.1 | Standard for tabular Q-learning |
| Discount factor (γ) | 0.95 | Long-term planning important in scheduling |
| Initial ε | 1.0 | Full exploration at start |
| ε decay | Linear to 0.01 over 300 episodes | Gradual exploitation shift |
| UCB exploration constant | √2 | Theoretical optimum |
| Boltzmann temperature | 1.0 → 0.1 | Decay over 500 episodes |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_environment.py -v
```

## Reproducing Results

To exactly reproduce the results in the paper:

```bash
# Set random seed for reproducibility
export PYTHONHASHSEED=42

# Run experiments
python -m src.experiments --seed 42 --episodes 500 --runs 10

# Generate all figures
python scripts/generate_figures.py --results results/experiment_results.pkl
```

## Extending the Project

### Adding a New Exploration Strategy

1. Add the strategy to `agent.py`:

```python
def select_action(self, state, episode):
    if self.exploration_strategy == 'your-strategy':
        # the implementation here
        return action
```

2. Update `experiments.py` to include the new strategy:

```python
strategies = ['epsilon-greedy', 'ucb', 'boltzmann', 'your-strategy']
```

### Modifying the Environment

Edit `src/environment.py` to change:
- Number of servers (`n_servers`)
- Server capacities (`server_capacity`)
- Job arrival rate (Poisson λ)
- Reward function

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rl-scheduling-2024,
  author = {Mohammed Aabed },
  title = {Reinforcement Learning for Dynamic Task Scheduling: 
           A Comparative Study of Exploration Strategies},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/mohammed-abed/rl-scheduling-benchmark}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Inspired by classical work on multi-armed bandits and exploration-exploitation trade-offs
- Environment design influenced by real-world data center scheduling challenges
- Statistical analysis follows best practices from [Dror et al. 2018](https://aclanthology.org/P18-1128/)

## Contact

Mohammed Aabed - maabed90@students.iugaza.edu.ps 

Project Link: [https://github.com/mohammed-abed/rl-scheduling-benchmark](https://github.com/mohammed-abed/rl-scheduling-benchmark)

## Future Work

- [ ] Extend to function approximation (Deep Q-Networks)
- [ ] Multi-agent scenarios with server cooperation
- [ ] Transfer learning across different job distributions
- [ ] Real-world validation with production traces
- [ ] Integrate with Ray RLlib for distributed training



## Development note: 

This project was built offline and pushed to GitHub upon completion rather than incrementally. I was based in Gaza during this period, where consistent internet access was not reliably available. The commit history does not reflect the actual development timeline."