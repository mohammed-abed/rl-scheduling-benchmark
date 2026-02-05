
"""
Q-Learning Agent with Multiple Exploration Strategies
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict


class QLearningAgent:
    """
    Q-Learning agent with pluggable exploration strategies.
    
    Supports:
        - Epsilon-greedy exploration
        - Upper Confidence Bound (UCB)
        - Boltzmann (softmax) exploration
    """
    
    def __init__(
        self,
        n_actions: int,
        exploration_strategy: str = 'epsilon-greedy',
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        random_seed: int = None
    ):
        """
        Initialize Q-learning agent.
        
        Args:
            n_actions: Number of possible actions
            exploration_strategy: One of ['epsilon-greedy', 'ucb', 'boltzmann']
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate (for epsilon-greedy)
            random_seed: Random seed for reproducibility
        """
        self.n_actions = n_actions
        self.exploration_strategy = exploration_strategy
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Visit counts for UCB
        self.visits = defaultdict(lambda: np.zeros(n_actions))
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def select_action(self, state: Tuple, episode: int = 0) -> int:
        """
        Select action according to exploration strategy.
        
        Args:
            state: Current state
            episode: Current episode number (used for some strategies)
            
        Returns:
            Action to take
        """
        if self.exploration_strategy == 'epsilon-greedy':
            return self._epsilon_greedy(state)
        elif self.exploration_strategy == 'ucb':
            return self._ucb(state)
        elif self.exploration_strategy == 'boltzmann':
            return self._boltzmann(state, episode)
        else:
            raise ValueError(f"Unknown strategy: {self.exploration_strategy}")
    
    def _epsilon_greedy(self, state: Tuple) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])
    
    def _ucb(self, state: Tuple) -> int:
        """Upper Confidence Bound action selection."""
        total_visits = np.sum(self.visits[state]) + 1
        
        # UCB formula: Q(s,a) + c * sqrt(ln(N) / n(s,a))
        ucb_values = self.q_table[state] + np.sqrt(
            2 * np.log(total_visits) / (self.visits[state] + 1)
        )
        
        return np.argmax(ucb_values)
    
    def _boltzmann(self, state: Tuple, episode: int) -> int:
        """Boltzmann (softmax) action selection with temperature decay."""
        # Temperature decay: 1.0 -> 0.1 over 500 episodes
        temperature = max(0.1, 1.0 - episode / 500)
        
        # Softmax with numerical stability
        q_values = self.q_table[state]
        exp_values = np.exp((q_values - np.max(q_values)) / temperature)
        probs = exp_values / np.sum(exp_values)
        
        return np.random.choice(self.n_actions, p=probs)
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> float:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            
        Returns:
            TD error (for debugging/monitoring)
        """
        # Update visit count (for UCB)
        self.visits[state][action] += 1
        
        # Q-learning update: Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.alpha * td_error
        
        return td_error
    
    def decay_epsilon(self, episode: int, total_episodes: int):
        """Decay epsilon linearly over training."""
        # Decay to 0.01 over 60% of training
        decay_period = int(total_episodes * 0.6)
        if episode < decay_period:
            self.epsilon = 1.0 - (episode / decay_period) * 0.99
        else:
            self.epsilon = 0.01
    
    def get_policy(self) -> Dict:
        """
        Extract greedy policy from Q-table.
        
        Returns:
            Dictionary mapping states to best actions
        """
        policy = {}
        for state in self.q_table.keys():
            policy[state] = np.argmax(self.q_table[state])
        return policy
    
    def get_value_function(self) -> Dict:
        """
        Extract value function from Q-table.
        
        Returns:
            Dictionary mapping states to state values
        """
        value_function = {}
        for state in self.q_table.keys():
            value_function[state] = np.max(self.q_table[state])
        return value_function
