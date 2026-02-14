from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .control_algorithms import epsilon_greedy_choice
from .utils import decay_schedule


def q_learning(
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

        while not done:
            action = choice_method(
                state, Q, episode, max_episodes=n_episodes, **choice_method_kwargs
            )
            next_state, reward, done, _, _ = env.step(action)

            td_target = reward + gamma * Q[next_state].max() * (not done)

            Q[state, action] += alphas[episode] * (td_target - Q[state, action])

            state = next_state

        Q_per_episode[episode] = Q
        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1
        policy_per_episode[episode] = policy

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, Q_per_episode, policy_per_episode


def double_q_learning(
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

    Q1 = np.zeros((n_states, n_actions), dtype=np.float32)
    Q2 = np.zeros((n_states, n_actions), dtype=np.float32)
    Q1_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.float32)
    Q2_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.float32)

    policy_per_episode = np.zeros((n_episodes, n_states, n_actions), dtype=np.int32)

    for episode in tqdm(range(n_episodes)):

        state, _ = env.reset()
        done = False

        while not done:
            action = choice_method(
                state,
                (Q1 + Q2) / 2,
                episode,
                max_episodes=n_episodes,
                **choice_method_kwargs,
            )
            next_state, reward, done, _, _ = env.step(action)

            if np.random.randint(2):
                argmax_Q1 = np.argmax(Q1[next_state])

                td_target = reward + gamma * Q2[next_state, argmax_Q1] * (not done)

                Q1[state, action] += alphas[episode] * (td_target - Q1[state, action])
            else:
                argmax_Q2 = np.argmax(Q2[next_state])

                td_target = reward + gamma * Q1[next_state, argmax_Q2] * (not done)

                Q2[state, action] += alphas[episode] * (td_target - Q2[state, action])

            state = next_state

        Q1_per_episode[episode] = Q1
        Q2_per_episode[episode] = Q2

        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax((Q1 + Q2) / 2, axis=1)] = 1
        policy_per_episode[episode] = policy

    Q = (Q1 + Q2) / 2

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, policy_per_episode


def q_lambda(
    env: Any,
    gamma: float = 1.0,
    lambda_val: float = 0.5,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 10_000,
    choice_method: Callable = epsilon_greedy_choice,
    replacing_traces: bool = True,
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

    traces = np.zeros((n_states, n_actions), dtype=np.float32)

    for episode in tqdm(range(n_episodes)):

        traces.fill(0)

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

            next_action_is_greedy = next_action == np.argmax(Q[next_state])

            td_target = reward + gamma * Q[next_state].max() * (not done)

            if replacing_traces:
                traces[state, action] = 0.0
            traces[state, action] += 1.0

            Q += alphas[episode] * (td_target - Q[state, action]) * traces

            if next_action_is_greedy:
                traces *= gamma * lambda_val
            else:
                traces.fill(0)

            state, action = next_state, next_action

        Q_per_episode[episode] = Q
        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1
        policy_per_episode[episode] = policy

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, Q_per_episode, policy_per_episode
