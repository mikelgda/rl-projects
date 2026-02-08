import numpy as np

from rl_projects.basic.policy_evaluation import policy_evaluation
from rl_projects.basic.policy_improvement import policy_improvement
from rl_projects.envs.frozen_lake import random_stochastic_policy


def policy_iteration(env, gamma=1.0, theta=1e-10):
    mdp = env.unwrapped.P
    policy = random_stochastic_policy(env)

    while True:
        old_policy = policy.copy()

        V = policy_evaluation(policy, mdp, gamma=gamma, theta=theta)

        policy = policy_improvement(V, mdp, gamma=gamma)

        if np.array_equal(policy, old_policy):
            break

    return V, policy


if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import seaborn as sns

    from rl_projects.envs.frozen_lake import OPTIMAL_FROZEN_LAKE_POLICY

    gamma = 0.9

    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        success_rate=1.0 / 3.0,
        reward_schedule=(1, 0, 0),
        render_mode="ansi",
    )

    iterated_V, iterated_policy = policy_iteration(env, gamma=gamma)
    optimal_V = policy_evaluation(
        OPTIMAL_FROZEN_LAKE_POLICY, env.unwrapped.P, gamma=gamma
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        iterated_V.reshape(4, 4), annot=True, cmap="viridis", cbar=False, ax=axes[0]
    )
    axes[0].set_title("Value Function of Iterated Policy")
    sns.heatmap(
        optimal_V.reshape(4, 4), annot=True, cmap="viridis", cbar=False, ax=axes[1]
    )
    axes[1].set_title("Value Function of Optimal Policy")
    plt.show()
