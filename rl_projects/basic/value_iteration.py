import numpy as np
from numpy.typing import NDArray


def value_iteration(
    mdp: dict[int, dict], gamma: float = 1.0, theta: float = 1e-10
) -> NDArray:
    V = np.zeros(len(mdp), dtype=np.float64)

    while True:
        Q = np.zeros((len(mdp), len(mdp[0])), dtype=np.float64)

        for state in mdp:
            for action in mdp[state]:
                for prob, next_state, reward, done in mdp[state][action]:
                    Q[state, action] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        V = np.max(Q, axis=1)

    policy = np.zeros_like(Q)
    policy[np.arange(len(Q)), np.argmax(Q, axis=1)] = 1

    return policy
