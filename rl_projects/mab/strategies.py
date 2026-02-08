from typing import Any, Callable

import numpy as np
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


def softmax_exploration(
    env: Any,
    n_episodes: int = 1000,
    init_temp: float = 1000.0,
    min_temp: float = 0.01,
    decay_ratio: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, N = zero_initialization(env.action_space.n)

    Q_per_event = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for episode in tqdm(range(n_episodes), leave=False):

        # linear decay from init_temp to min_temp over decay_ratio of the episodes
        decay_episodes = n_episodes * decay_ratio
        temp = 1 - episode / decay_episodes
        temp *= init_temp - min_temp
        temp += min_temp
        temp = np.clip(temp, min_temp, init_temp)

        scaled_Q = Q / temp
        norm_Q = scaled_Q - np.max(scaled_Q)
        exp_Q = np.exp(norm_Q)
        probs = exp_Q / np.sum(exp_Q)

        assert np.isclose(np.sum(probs), 1.0), f"Probabilities do not sum to 1: {probs}"
        action = np.random.choice(np.arange(len(Q)), p=probs, size=1).item()

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


def ucb_exploration(
    env: Any, confidence_constant: int = 2, n_episodes: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, N = zero_initialization(env.action_space.n)

    Q_per_event = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for episode in tqdm(range(n_episodes), leave=False):
        if episode < len(Q):
            action = episode
        else:

            U = np.sqrt((confidence_constant * np.log(episode)) / N)
            action = np.argmax(Q + U)

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


def gaussian_thompson_sampling(
    env: Any, alpha: float = 1.0, beta: float = 0.0, n_episodes: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, N = zero_initialization(env.action_space.n)

    Q_per_event = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    for episode in tqdm(range(n_episodes), leave=False):
        samples = np.random.normal(loc=Q, scale=alpha / (np.sqrt(N) + beta))

        action = np.argmax(samples)

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


def beta_thompson_sampling(
    env: Any, n_episodes: int = 1000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Initialize Alpha (successes) and Beta (failures) to 1 for a uniform prior
    alphas = np.ones(env.action_space.n)
    betas = np.ones(env.action_space.n)

    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    Q_per_event = np.empty((n_episodes, env.action_space.n))

    for episode in tqdm(range(n_episodes), leave=False):
        # 1. SAMPLE: Draw from the Beta distribution for each arm
        # This replaces the np.random.normal line
        samples = np.random.beta(alphas, betas)

        # 2. SELECT: Pick the arm with the highest sample
        action = np.argmax(samples)

        # 3. OBSERVE: Get reward (assumes reward is 0 or 1)
        _, reward, _, _, _ = env.step(action)

        # 4. UPDATE: Bayesian update of the Posterior
        if reward == 1:
            alphas[action] += 1
        else:
            betas[action] += 1

        returns[episode] = reward
        actions[episode] = action
        Q_per_event[episode] = alphas / (alphas + betas)
    return returns, Q_per_event, actions
