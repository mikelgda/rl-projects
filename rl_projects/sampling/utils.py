from typing import Callable

import numpy as np


def collect_trajectory_from_policy(env, policy, max_steps=20):
    trajectory = []
    done = False
    while not done:
        state, _ = env.reset()
        for t in range(max_steps):
            action = np.random.multinomial(n=1, pvals=policy[state]).argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            experience = (state, action, reward, next_state, done, truncated)
            trajectory.append(experience)

            done = done or truncated
            if done:
                break
            if t >= max_steps - 1:
                trajectory = []
                break
            state = next_state

    return np.array(trajectory, dtype=object)


def collect_trajectory_from_Q_function(
    Q: np.ndarray,
    env,
    choice_method: Callable,
    episode: int = 0,
    max_steps: int = 200,
    **choice_method_kwargs,
):
    done = False
    trajectory = []

    while not done:
        state, _ = env.reset()

        for i in range(max_steps):

            action = choice_method(state, Q, episode=episode, **choice_method_kwargs)
            next_state, reward, done, truncated, _ = env.step(action)
            experience = (state, action, reward, next_state, done, truncated)

            trajectory.append(experience)

            if done or truncated:
                break

            if i >= max_steps - 1:
                trajectory = []
                break

            state = next_state

    return np.array(trajectory, dtype=object)


def decay_schedule(
    initial_value, min_value, decay_rate, max_steps, log_start=-2, log_base=10
):
    decay_steps = int(max_steps * decay_rate)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]

    values = (values - values.min()) / (values.max() - values.min())

    values = (initial_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), "edge")

    return values
