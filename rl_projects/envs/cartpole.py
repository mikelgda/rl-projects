import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display


def get_default_env(render_mode="rgb_array"):
    return gym.make("CartPole-v1", render_mode=render_mode)


def random_policy(state):
    return np.random.randint(2)


def render_episode(env, policy, max_steps=500, ax=None):

    if env.render_mode != "rgb_array":
        raise ValueError(
            "Environment must be initialized with render_mode='rgb_array' to use this function."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.axis("off")
    im = None

    state, _ = env.reset()

    for i in range(max_steps):  # Run for 100 steps
        action = policy(state)
        state, _, terminated, truncated, _ = env.step(action)
        image = env.render()

        if im is None:
            im = ax.imshow(image)
        else:
            im.set_data(image)

        ax.set_title(f"Step: {i+1}")

        clear_output(wait=True)
        display(fig)

        if terminated or truncated:
            state, _ = env.reset()
