from typing import Tuple, Type

import numpy as np
import torch.nn as nn
from numpy.typing import ArrayLike


def get_tqdm():
    try:
        # Check if we are in an IPython environment
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            from tqdm.notebook import tqdm

            return tqdm
        else:
            from tqdm import tqdm

            return tqdm
    except NameError:
        # Not in IPython/Jupyter at all
        from tqdm import tqdm

        return tqdm


# Assign it to a variable for use throughout your class
safe_tqdm = get_tqdm()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (32, 32),
        activation: Type[nn.Module] = nn.ReLU,
    ):
        """
        Multi-layer perceptron with variable hidden layers and dimensions.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function class (default: ReLU)
        """
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EpsilonGreedyChoice:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def choose_action(self, q_values: ArrayLike) -> int:
        """
        Select an action using epsilon-greedy strategy.

        Args:
            q_values: Tensor of shape (action_space_size,) containing Q-values for each action

        Returns:
            Selected action index
        """
        q_values = np.array(q_values)  # Ensure q_values is a numpy array

        random_number = np.random.rand(1).item()

        if random_number < self.epsilon:
            # Explore: choose a random action
            return int(np.random.randint(0, q_values.shape[-1], 1))
        else:
            # Exploit: choose the action with the highest Q-value
            return int(np.argmax(q_values))
