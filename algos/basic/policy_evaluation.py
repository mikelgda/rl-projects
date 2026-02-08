import numpy as np


def policy_evaluation(policy, mdp, gamma=1.0, theta=1e-10):
    """
    Evaluate a policy in a Markov Decision Process (MDP).

    Args:
        policy: A policy function that maps states to actions.
        mdp: A Markov Decision Process (MDP) represented as a dictionary.
        gamma: The discount factor.
        theta: The convergence threshold.

    Returns:
        The value function for the given policy.
    """
    n_states = len(mdp)
    n_actions = len(mdp[0])
    assert policy.shape == (
        n_states,
        n_actions,
    ), "Policy shape must match the number of states and actions in the MDP."

    last_V = np.zeros(n_states)

    while True:
        V = np.zeros(n_states)

        for state in range(n_states):
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in mdp[state][action]:
                    V[state] += (
                        action_prob
                        * prob
                        * (reward + gamma * last_V[next_state] * (not done))
                    )

        if np.max(np.abs(V - last_V)) < theta:
            break

        last_V = V.copy()

    return V


if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils import OPTIMAL_FROZEN_LAKE_POLICY, random_stochastic_policy

    env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        success_rate=1.0 / 3.0,
        reward_schedule=(1, 0, 0),
        render_mode="ansi",
    )

    mdp = env.unwrapped.P

    random_policy = random_stochastic_policy(env)

    gamma = 0.9

    random_V = policy_evaluation(random_policy, mdp, gamma=gamma)
    optimal_V = policy_evaluation(OPTIMAL_FROZEN_LAKE_POLICY, mdp, gamma=gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(optimal_V.reshape(4, 4), cmap="viridis", annot=True, ax=axes[0])
    sns.heatmap(random_V.reshape(4, 4), cmap="viridis", annot=True, ax=axes[1])
    axes[0].set_title("Optimal Policy Value Function")
    axes[1].set_title("Random Policy Value Function")

    plt.show()
