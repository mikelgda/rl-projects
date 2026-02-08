import numpy as np
from numpy.typing import NDArray

from rl_projects.envs.frozen_lake import random_stochastic_policy


def compute_Q_function(V: NDArray, mdp: dict[int, dict], gamma: float = 1.0) -> NDArray:
    Q = np.zeros((len(mdp), len(mdp[0])), dtype=np.float64)

    for state in range(len(mdp)):
        for action in range(len(mdp[state])):
            for prob, next_state, reward, done in mdp[state][action]:
                Q[state, action] += prob * (reward + gamma * V[next_state] * (not done))

    return Q


def policy_improvement(V: NDArray, mdp: dict[int, dict], gamma: float = 1.0) -> NDArray:
    Q = compute_Q_function(V, mdp, gamma)
    new_Q = np.zeros_like(Q)
    new_Q[np.arange(len(Q)), np.argmax(Q, axis=1)] = 1
    return new_Q


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    from rl_projects.basic.policy_evaluation import policy_evaluation
    from rl_projects.envs.frozen_lake import get_default_env

    env = get_default_env()

    mdp = env.unwrapped.P  # type: ignore

    gamma = 0.9

    random_policy = random_stochastic_policy(env)
    random_V = policy_evaluation(random_policy, mdp, gamma=gamma)
    new_random_policy = policy_improvement(random_V, mdp, gamma=gamma)
    new_random_V = policy_evaluation(new_random_policy, mdp, gamma=gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        random_V.reshape(4, 4), annot=True, cmap="viridis", cbar=False, ax=axes[0]
    )
    axes[0].set_title("Value Function of Random Policy")
    sns.heatmap(
        new_random_V.reshape(4, 4), annot=True, cmap="viridis", cbar=False, ax=axes[1]
    )
    axes[1].set_title("Value Function of Improved Policy")

    plt.show()
