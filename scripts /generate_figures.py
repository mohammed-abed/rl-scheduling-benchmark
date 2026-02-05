"""
Generate all figures from experimental results
"""

import argparse
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description='Generate figures from results')
    parser.add_argument('--results', type=str, default='results/experiment_results.pkl',
                       help='Path to results pickle file')
    parser.add_argument('--output', type=str, default='results/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    
    # Generate figures
    generate_all_figures(results, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
```

**scripts/run_experiments.sh**
```bash
#!/bin/bash

# Run full experimental suite

echo "Running RL Scheduling Benchmark Experiments"
echo "==========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run experiments
python -m src.experiments \
    --episodes 500 \
    --runs 10 \
    --seed 42 \
    --output results

# Generate figures
python scripts/generate_figures.py \
    --results results/experiment_results.pkl \
    --output results/figures

echo ""
echo "Experiments complete!"
echo "Results saved in: results/"
echo "Figures saved in: results/figures/"

