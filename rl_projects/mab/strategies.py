from typing import Callable

import numpy as np
from matplotlib.pylab import Any
from tqdm import tqdm


def random_initialization(n_actions: int) -> tuple[np.ndarray, np.ndarray]:
    Q = np.random.rand(n_actions)  # Random initialization to break ties
    N = np.zeros(n_actions)
    return Q, N


def zero_initialization(n_actions: int) -> tuple[np.ndarray, np.ndarray]:
    Q = np.zeros(n_actions)  # Zero initialization
    N = np.zeros(n_actions)
    return Q, N


def optimistic_initialization(
    n_actions: int, optimistic_estimate: float = 1.0, initial_count: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    Q = np.full(
        n_actions, optimistic_estimate, dtype=np.float64
    )  # Optimistic initialization
    N = np.full(n_actions, initial_count, dtype=np.float64)
    return Q, N


def pure_exploitation(
    env: Any,
    n_episodes: int = 1000,
    initialization: Callable = optimistic_initialization,
    **init_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, N = initialization(env.action_space.n, **init_kwargs)

    Q_per_event = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for episode in tqdm(range(n_episodes)):
        action = np.argmax(Q)

        (
            _,
            reward,
            _,
            _,
            _,
        ) = env.step(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        Q_per_event[episode] = Q
        returns[episode] = reward
        actions[episode] = action

    return returns, Q_per_event, actions


def pure_exploration(
    env: Any, n_episodes: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, N = zero_initialization(env.action_space.n)

    Q_per_event = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for episode in tqdm(range(n_episodes), leave=False):
        action = np.random.randint(len(Q))

        (
            _,
            reward,
            _,
            _,
            _,
        ) = env.step(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        Q_per_event[episode] = Q
        returns[episode] = reward
        actions[episode] = action

    return returns, Q_per_event, actions


def linear_decay(
    episode: int,
    initial_epsilon: float = 0.1,
    n_episodes: int = 1000,
    min_epsilon: float = 0.01,
    decay_ratio: float = 0.05,
) -> float:
    decay_episodes = int(n_episodes * decay_ratio)

    epsilon = 1 - episode / decay_episodes
    epsilon *= initial_epsilon - min_epsilon
    epsilon += min_epsilon

    epsilon = np.clip(epsilon, min_epsilon, initial_epsilon)

    return epsilon


def epsilon_greedy(
    env: Any,
    epsilon: float = 0.01,
    n_episodes: int = 1000,
    decay_function: Callable | None = None,
    **decay_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, N = zero_initialization(env.action_space.n)

    Q_per_event = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for episode in tqdm(range(n_episodes), leave=False):
        if decay_function is not None:
            epsilon = decay_function(episode, **decay_kwargs)
        if np.random.rand() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        (
            _,
            reward,
            _,
            _,
            _,
        ) = env.step(action)
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        Q_per_event[episode] = Q
        returns[episode] = reward
        actions[episode] = action

    return returns, Q_per_event, actions
