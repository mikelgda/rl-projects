from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .utils import (
    collect_trajectory_from_policy,
    collect_trajectory_from_Q_function,
    decay_schedule,
)


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


def monte_carlo_control(
    env: Any,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 10_000,
    max_steps: int = 200,
    first_visit: bool = True,
    choice_method: Callable = epsilon_greedy_choice,
    **choice_method_kwargs,
):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    gamma_discount = np.logspace(
        0, max_steps, num=max_steps, base=gamma, endpoint=False
    )

    alphas = decay_schedule(
        initial_value=initial_alpha,
        min_value=min_alpha,
        decay_rate=alpha_decay_rate,
        max_steps=n_episodes,
    )

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    Q_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.float32)

    policy_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.int32)

    for episode in tqdm(range(n_episodes)):

        trajectory = collect_trajectory_from_Q_function(
            Q,
            env,
            choice_method=choice_method,
            max_steps=max_steps,
            episode=episode,
            max_episodes=n_episodes,
            **choice_method_kwargs,
        )

        visited = np.zeros((n_states, n_actions), dtype=bool)

        for t, (state, action, _, _, _, _) in enumerate(trajectory):
            if visited[state, action] and first_visit:
                continue
            visited[state, action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(gamma_discount[:n_steps] * trajectory[t:, 2])
            Q[state, action] += alphas[episode] * (G - Q[state, action])

        Q_per_episode[episode] = Q

        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1
        policy_per_episode[episode] = policy

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, Q_per_episode, policy_per_episode


def sarsa(
    env: Any,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 10_000,
    choice_method: Callable = epsilon_greedy_choice,
    **choice_method_kwargs,
):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    alphas = decay_schedule(
        initial_value=initial_alpha,
        min_value=min_alpha,
        decay_rate=alpha_decay_rate,
        max_steps=n_episodes,
    )

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    Q_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.float32)

    policy_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.int32)

    for episode in tqdm(range(n_episodes)):

        state, _ = env.reset()
        done = False
        action = choice_method(
            state, Q, episode, max_episodes=n_episodes, **choice_method_kwargs
        )

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = choice_method(
                next_state, Q, episode, max_episodes=n_episodes, **choice_method_kwargs
            )

            td_target = reward + gamma * Q[next_state, next_action] * (not done)

            Q[state, action] += alphas[episode] * (td_target - Q[state, action])

            state, action = next_state, next_action

        Q_per_episode[episode] = Q
        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1
        policy_per_episode[episode] = policy

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, Q_per_episode, policy_per_episode
