import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BanditSlipperyWalk(gym.Env):
    def __init__(self, slip_prob=0.2):
        super(BanditSlipperyWalk, self).__init__()

        self.n_states = 3
        self.slip_prob = slip_prob
        self.start_state = 1  # Start at the leftmost state
        self.goal_state = 2
        self.fail_state = 0
        self.current_state = self.start_state

        # Action 0: Left, Action 1: Right
        self.action_space = spaces.Discrete(2)

        # Observation is the current position on the walk
        self.observation_space = spaces.Discrete(self.n_states)

    def step(self, action):
        # Apply the "Slippery" logic (The Bandit uncertainty)
        if np.random.random() < self.slip_prob:
            action = 1 - action  # Flip the action (0 becomes 1, 1 becomes 0)

        # Update state based on action
        if action == 1:  # Move Right
            self.current_state = min(self.goal_state, self.current_state + 1)
        else:  # Move Left
            self.current_state = max(0, self.current_state - 1)

        # Check if goal is reached
        terminated = (self.current_state == self.goal_state) or (
            self.current_state == self.fail_state
        )
        reward = 1.0 if (self.current_state == self.goal_state) else 0.0
        truncated = False  # For compatibility

        return self.current_state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed, options=options, **kwargs)
        self.current_state = self.start_state
        return self.current_state, {}


class BernoulliBandit(gym.Env):
    def __init__(self, alpha=0.1, beta=0.1):
        super(BernoulliBandit, self).__init__()

        self.n_states = 3
        self.alpha = alpha
        self.beta = beta
        self.start_state = 1  # Start at the leftmost state
        self.goal_state = 2
        self.fail_state = 0
        self.current_state = self.start_state

        # Action 0: Left, Action 1: Right
        self.action_space = spaces.Discrete(2)

        # Observation is the current position on the walk
        self.observation_space = spaces.Discrete(self.n_states)

    def step(self, action):

        random_number = np.random.random()  # Update state based on action
        if action == 1:  # Move Right
            if random_number < self.alpha:
                self.current_state = max(0, self.current_state - 1)
            else:
                self.current_state = min(self.goal_state, self.current_state + 1)
        else:  # Move Left
            if random_number < self.beta:
                self.current_state = min(self.goal_state, self.current_state + 1)
            else:
                self.current_state = max(0, self.current_state - 1)

        # Check if goal is reached
        terminated = (self.current_state == self.goal_state) or (
            self.current_state == self.fail_state
        )
        reward = 1.0 if (self.current_state == self.goal_state) else 0.0
        truncated = False  # For compatibility

        return self.current_state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed, options=options, **kwargs)
        self.current_state = self.start_state
        return self.current_state, {}


class SimpleBandit(gym.Env):
    def __init__(self, k=10):
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Discrete(1)  # Stateless
        self.means = np.random.normal(0, 1, k)

    def step(self, action):
        reward = np.random.normal(self.means[action], 1.0)
        return 0, reward, False, False, {}

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed, options=options, **kwargs)
        return 0, {}
