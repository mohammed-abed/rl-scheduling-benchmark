"""Unit tests for QLearningAgent"""

import pytest
import numpy as np
from src.agent import QLearningAgent


def test_agent_initialization():
    """Test agent initializes correctly."""
    agent = QLearningAgent(n_actions=5, random_seed=42)
    
    assert agent.n_actions == 5
    assert agent.alpha == 0.1
    assert agent.gamma == 0.95
    assert agent.epsilon == 1.0


def test_epsilon_greedy():
    """Test epsilon-greedy action selection."""
    agent = QLearningAgent(n_actions=5, exploration_strategy='epsilon-greedy', random_seed=42)
    
    state = (0, (0, 0, 0, 0, 0), 0)
    
    # With epsilon=1.0, should explore
    agent.epsilon = 1.0
    actions = [agent.select_action(state) for _ in range(100)]
    assert len(set(actions)) > 1  # Should have variety
    
    # With epsilon=0.0, should always choose best action
    agent.epsilon = 0.0
    agent.q_table[state][2] = 10.0  # Make action 2 clearly best
    actions = [agent.select_action(state) for _ in range(100)]
    assert all(a == 2 for a in actions)


def test_ucb():
    """Test UCB action selection."""
    agent = QLearningAgent(n_actions=5, exploration_strategy='ucb', random_seed=42)
    
    state = (0, (0, 0, 0, 0, 0), 0)
    
    # Should explore all actions initially
    actions = [agent.select_action(state) for _ in range(100)]
    # Update visits as if we took those actions
    for a in actions:
        agent.visits[state][a] += 1
    
    # After many visits, should converge
    assert len(set(actions)) == 5  # All actions tried


def test_boltzmann():
    """Test Boltzmann action selection."""
    agent = QLearningAgent(n_actions=5, exploration_strategy='boltzmann', random_seed=42)
    
    state = (0, (0, 0, 0, 0, 0), 0)
    
    # With high temperature (early episodes), more random
    actions_early = [agent.select_action(state, episode=0) for _ in range(100)]
    
    # With low temperature (late episodes), more greedy
    agent.q_table[state][2] = 10.0
    actions_late = [agent.select_action(state, episode=500) for _ in range(100)]
    
    # Late episodes should prefer action 2
    assert actions_late.count(2) > actions_early.count(2)


def test_update():
    """Test Q-value update."""
    agent = QLearningAgent(n_actions=5, random_seed=42)
    
    state = (0, (0, 0, 0, 0, 0), 0)
    next_state = (0, (0, 0, 0, 0, 0), 1)
    action = 2
    reward = 10.0
    
    old_q = agent.q_table[state][action]
    td_error = agent.update(state, action, reward, next_state)
    new_q = agent.q_table[state][action]
    
    assert new_q != old_q
    assert isinstance(td_error, float)


def test_epsilon_decay():
    """Test epsilon decay."""
    agent = QLearningAgent(n_actions=5, exploration_strategy='epsilon-greedy')
    
    assert agent.epsilon == 1.0
    
    agent.decay_epsilon(episode=0, total_episodes=500)
    assert agent.epsilon == 1.0
    
    agent.decay_epsilon(episode=150, total_episodes=500)
    assert 0.01 < agent.epsilon < 1.0
    
    agent.decay_epsilon(episode=500, total_episodes=500)
    assert agent.epsilon == 0.01


def test_reproducibility():
    """Test that same seed produces same actions."""
    agent1 = QLearningAgent(n_actions=5, random_seed=42)
    agent2 = QLearningAgent(n_actions=5, random_seed=42)
    
    state = (0, (0, 0, 0, 0, 0), 0)
    
    actions1 = [agent1.select_action(state) for _ in range(100)]
    actions2 = [agent2.select_action(state) for _ in range(100)]
    
    assert actions1 == actions2
