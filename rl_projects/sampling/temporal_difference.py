from typing import Any

import numpy as np
from tqdm import tqdm

from .utils import collect_trajectory, decay_schedule


def td_policy_evaluation(
    env: Any,
    policy: np.ndarray,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.3,
    n_episodes: int = 500,
):

    n_states = env.observation_space.n

    alphas = decay_schedule(
        initial_value=initial_alpha,
        min_value=min_alpha,
        decay_rate=alpha_decay_rate,
        max_steps=n_episodes,
    )

    V = np.zeros(n_states)
    V_per_episode = np.zeros((n_episodes, n_states))

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.random.multinomial(n=1, pvals=policy[state]).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            td_target = reward + gamma * V[next_state] * (not done)
            V[state] += alphas[episode] * (td_target - V[state])
            V_per_episode[episode] = V

            state = next_state

    return V, V_per_episode


def tdn_policy_evaluation(
    env: Any,
    policy: np.ndarray,
    gamma: float = 1.0,
    initial_alpha: float = 0.5,
    min_alpha: float = 0.01,
    alpha_decay_rate: float = 0.5,
    n_steps: int = 3,
    n_episodes: int = 500,
):

    n_states = env.observation_space.n

    alphas = decay_schedule(
        initial_value=initial_alpha,
        min_value=min_alpha,
        decay_rate=alpha_decay_rate,
        max_steps=n_episodes,
    )

    gamma_discount = np.logspace(
        0, n_steps + 1, num=n_steps + 1, base=gamma, endpoint=False
    )

    V = np.zeros(n_states)
    V_per_episode = np.zeros((n_episodes, n_states))

    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        done = False
        path = []
        while not done or (path is not None):
            path = path[1:]

            while not done and len(path) < n_steps:
                action = np.random.multinomial(n=1, pvals=policy[state]).argmax().item()

                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated

                path.append((state, action, reward, done))

                state = next_state
                if done:
                    break

            n = len(path)
            est_state = path[0][0]

            rewards = np.array(path)[:, 2]
            partial_return = gamma_discount[:n] * rewards

            bs_val = gamma_discount[-1] * V[next_state] * (not done)

            tdn_target = np.sum(np.append(partial_return, bs_val))

            V[est_state] += alphas[episode] * (tdn_target - V[est_state])

            if len(path) == 1 and path[0][3]:
                path = None

            V_per_episode[episode] = V

    return V, V_per_episode
