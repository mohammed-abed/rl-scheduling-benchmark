"""Unit tests for ResourceSchedulingEnv"""

import pytest
import numpy as np
from src.environment import ResourceSchedulingEnv


def test_environment_initialization():
    """Test environment initializes correctly."""
    env = ResourceSchedulingEnv(n_servers=5, max_queue=20, random_seed=42)
    
    assert env.n_servers == 5
    assert env.max_queue == 20
    assert len(env.server_capacity) == 5
    assert np.all(env.server_loads == 0)
    assert len(env.queue) == 0


def test_reset():
    """Test reset functionality."""
    env = ResourceSchedulingEnv(random_seed=42)
    
    # Run some steps
    for _ in range(10):
        env.step(0)
    
    # Reset
    state = env.reset()
    
    assert np.all(env.server_loads == 0)
    assert len(env.queue) == 0
    assert env.time == 0
    assert len(env.completed_jobs) == 0
    assert state is not None


def test_step():
    """Test step function."""
    env = ResourceSchedulingEnv(random_seed=42)
    state = env.reset()
    
    next_state, reward, done, info = env.step(0)
    
    assert next_state is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'queue_length' in info


def test_state_space():
    """Test state representation."""
    env = ResourceSchedulingEnv(random_seed=42)
    state = env.reset()
    
    assert isinstance(state, tuple)
    assert len(state) == 3  # queue_state, load_state, priority_state
    assert isinstance(state[0], (int, np.integer))
    assert isinstance(state[1], tuple)
    assert isinstance(state[2], (int, np.integer))


def test_reproducibility():
    """Test that same seed produces same results."""
    env1 = ResourceSchedulingEnv(random_seed=42)
    env2 = ResourceSchedulingEnv(random_seed=42)
    
    env1.reset()
    env2.reset()
    
    for _ in range(100):
        s1, r1, d1, _ = env1.step(0)
        s2, r2, d2, _ = env2.step(0)
        
        assert s1 == s2
        assert r1 == r2
        assert d1 == d2


def test_queue_overflow():
    """Test queue overflow handling."""
    env = ResourceSchedulingEnv(n_servers=1, max_queue=5, random_seed=42)
    env.reset()
    
    # Force queue to fill
    for _ in range(100):
        env.step(0)  # Don't assign jobs
    
    assert len(env.queue) <= env.max_queue


def test_episode_termination():
    """Test episode terminates correctly."""
    env = ResourceSchedulingEnv(random_seed=42)
    env.reset()
    
    done = False
    steps = 0
    
    while not done and steps < 2000:
        _, _, done, _ = env.step(np.random.randint(env.n_servers))
        steps += 1
    
    assert done
    assert steps <= 1000  # Should terminate at 1000 steps
