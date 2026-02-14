from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .control_algorithms import epsilon_greedy_choice
from .utils import decay_schedule


def dyna_q(
    env: Any,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 10_000,
    n_planning_steps: int = 3,
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

    visit_counts = np.zeros((n_states, n_actions, n_states), dtype=np.int32)
    reward_model = np.zeros((n_states, n_actions, n_states), dtype=np.float32)

    for episode in tqdm(range(n_episodes)):

        state, _ = env.reset()
        done = False

        while not done:
            action = choice_method(
                state, Q, episode, max_episodes=n_episodes, **choice_method_kwargs
            )
            next_state, reward, done, _, _ = env.step(action)

            visit_counts[state, action, next_state] += 1

            reward_delta = reward - reward_model[state, action, next_state]

            reward_model[state, action, next_state] += (
                reward_delta / visit_counts[state, action, next_state]
            )

            td_target = reward + gamma * Q[next_state].max() * (not done)

            Q[state, action] += alphas[episode] * (td_target - Q[state, action])

            # planning steps
            for _ in range(n_planning_steps):
                if Q.sum() == 0:
                    break

                visited_states = np.where(np.sum(visit_counts, axis=(1, 2)) > 0)[0]
                state = np.random.choice(visited_states)

                actions_taken = np.where(np.sum(visit_counts[state], axis=1) > 0)[0]
                action = np.random.choice(actions_taken)

                transition_probs = (
                    visit_counts[state, action, :] / visit_counts[state, action].sum()
                )
                next_planning_state = np.random.choice(n_states, p=transition_probs)

                reward = reward_model[state, action, next_planning_state]
                td_target = reward + gamma * Q[next_planning_state].max()

                Q[state, action] += alphas[episode] * (td_target - Q[state, action])

            state = next_state

        Q_per_episode[episode] = Q
        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1
        policy_per_episode[episode] = policy

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, Q_per_episode, policy_per_episode


def trajectory_sampling(
    env: Any,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 10_000,
    max_planning_steps: int = 100,
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

    visit_counts = np.zeros((n_states, n_actions, n_states), dtype=np.int32)
    reward_model = np.zeros((n_states, n_actions, n_states), dtype=np.float32)

    for episode in tqdm(range(n_episodes)):

        state, _ = env.reset()
        done = False

        while not done:
            action = choice_method(
                state, Q, episode, max_episodes=n_episodes, **choice_method_kwargs
            )
            next_state, reward, done, _, _ = env.step(action)

            visit_counts[state, action, next_state] += 1

            reward_delta = reward - reward_model[state, action, next_state]

            reward_model[state, action, next_state] += (
                reward_delta / visit_counts[state, action, next_state]
            )

            td_target = reward + gamma * Q[next_state].max() * (not done)

            Q[state, action] += alphas[episode] * (td_target - Q[state, action])

            # planning steps
            for _ in range(max_planning_steps):
                if Q.sum() == 0:
                    break

                action = choice_method(
                    state, Q, episode, max_episodes=n_episodes, **choice_method_kwargs
                )

                if visit_counts[state, action].sum() == 0:
                    break

                transition_probs = (
                    visit_counts[state, action, :] / visit_counts[state, action].sum()
                )
                next_planning_state = np.random.choice(n_states, p=transition_probs)

                reward = reward_model[state, action, next_planning_state]

                td_target = reward + gamma * Q[next_planning_state].max()
                Q[state, action] += alphas[episode] * (td_target - Q[state, action])

            state = next_state

        Q_per_episode[episode] = Q
        policy = np.zeros((n_states, n_actions), dtype=int)
        policy[np.arange(n_states), np.argmax(Q, axis=1)] = 1
        policy_per_episode[episode] = policy

    V = np.max(Q, axis=1)
    policy = policy_per_episode[-1]

    return Q, V, policy, Q_per_episode, policy_per_episode
