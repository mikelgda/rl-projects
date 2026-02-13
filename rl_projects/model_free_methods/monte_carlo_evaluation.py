from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .utils import collect_trajectory_from_policy, decay_schedule


def monte_carlo_policy_evaluation(
    env: Any,
    policy: np.ndarray,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 500,
    max_steps: int = 100,
    first_visit: bool = True,
):
    n_states = env.observation_space.n

    gamma_discount = np.logspace(
        0, max_steps, num=max_steps, base=gamma, endpoint=False
    )

    alphas = decay_schedule(
        initial_value=initial_alpha,
        min_value=min_alpha,
        decay_rate=alpha_decay_rate,
        max_steps=n_episodes,
    )

    V = np.zeros(n_states)
    V_per_episode = np.zeros((n_episodes, n_states))

    for episode in tqdm(range(n_episodes)):
        trajectory = collect_trajectory_from_policy(env, policy, max_steps)

        visited = np.zeros(n_states, dtype=bool)

        for t, (state, _, _, _, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True
            n_steps = len(trajectory[t:])
            G = np.sum(gamma_discount[:n_steps] * trajectory[t:, 2])
            V[state] += alphas[episode] * (G - V[state])

        V_per_episode[episode] = V

    return V, V_per_episode


def epsilon_greedy_choice(
    state,
    Q,
    episode,
    max_episodes=10_000,
    initial_epsilon=1.0,
    min_epsilon=0.01,
    decay_rate=0.9,
):
    epsilon = decay_schedule(
        initial_value=initial_epsilon,
        min_value=min_epsilon,
        decay_rate=decay_rate,
        max_steps=max_episodes,
    )[episode]

    if np.random.random() > epsilon:
        return np.argmax(Q[state]).item()
    else:
        return np.random.randint(Q.shape[1])
