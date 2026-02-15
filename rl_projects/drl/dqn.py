import gc
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import MLP, ActionChoice, EpsilonGreedyChoice, ReplayBuffer, safe_tqdm


class DQNAgent:

    def __init__(
        self,
        env,
        policy_factory: Callable[[], nn.Module],
        action_choice: ActionChoice = EpsilonGreedyChoice(),
        gamma: float = 1.0,
        target_update_freq: int = 100,
        optimizer=torch.optim.RMSprop,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        loss_fn: nn.Module = nn.HuberLoss(),
        replay_buffer_size: int = 10000,
        n_batches_per_sample: int = 3,
    ) -> None:
        self.policy_model = policy_factory().to(device)
        self.target_model = policy_factory().to(device)
        self.device = device
        self.action_choice = action_choice
        self.n_actions = env.action_space.n
        self.obs_dim = env.observation_space.shape[0]
        self.gamma = gamma
        self.optimizer = optimizer(self.policy_model.parameters(), lr=0.0005)
        self.target_update_freq = target_update_freq
        self.loss_fn = loss_fn
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.n_batches_per_sample_round = n_batches_per_sample

    def interact_with_env(self, state: np.ndarray, env) -> Tuple[np.ndarray, bool]:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.policy_model(state_tensor).cpu().numpy()
        action = self.action_choice.choose_action(q_values)  # Example epsilon value

        next_state, reward, done, truncated, _ = env.step(action)

        failed = done and not truncated

        experience = (state, action, reward, next_state, int(failed))
        self.replay_buffer.store_experience(experience)

        return next_state, done

    def train(self, env, n_episodes: int = 3000, n_epochs: int = 40) -> None:
        time_step = 1
        sampled_estimates = 0
        for episode in safe_tqdm(range(n_episodes), desc="Episode"):
            state, _ = env.reset()

            while True:
                state, done = self.interact_with_env(state, env)
                sampled_estimates += 1

                if sampled_estimates >= (
                    self.n_batches_per_sample_round * self.replay_buffer.batch_size
                ):
                    experiences = self.replay_buffer.sample_batch()
                    experiences = self.process_experiences(experiences)
                    for _ in range(n_epochs):
                        self.optimize_model(experiences)

                if time_step % self.target_update_freq == 0:
                    self._update_target_model()

                if done:
                    gc.collect()
                    break

    def optimize_model(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        states, actions, rewards, next_states, fails = experiences
        max_q = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)

        q_values = self.policy_model(states).gather(1, actions)

        td_targets = rewards + self.gamma * max_q * (1 - fails)

        loss = self.loss_fn(q_values, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process_experiences(
        self, experiences
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, fails = experiences
        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )
        fails = torch.tensor(
            fails,
            dtype=torch.float32,
            device=self.device,
        )

        return states, actions, rewards, next_states, fails

    def _update_target_model(self) -> None:
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def __call__(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_model(state_tensor)
            action = q_values.argmax().item()

        return action
