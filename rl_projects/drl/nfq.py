import gc

import numpy as np
import torch
import torch.nn as nn

from .utils import MLP, EpsilonGreedyChoice, safe_tqdm


class NFQAgent:

    def __init__(
        self,
        env,
        policy_model,
        action_choice=EpsilonGreedyChoice(),
        epsilon=0.5,
        gamma=1.0,
        optimizer=torch.optim.RMSprop,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.policy_model = policy_model.to(device)
        self.device = device
        self.action_choice = action_choice
        self.n_actions = env.action_space.n
        self.obs_dim = env.observation_space.shape[0]
        self.epsilon = epsilon
        self.experiences = []
        self.gamma = gamma
        self.optimizer = optimizer(self.policy_model.parameters(), lr=0.0005)

    def interact_with_env(self, state, env):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.policy_model(state_tensor).cpu().numpy()
        action = self.action_choice.choose_action(q_values)  # Example epsilon value

        next_state, reward, done, truncated, _ = env.step(action)

        failed = done and not truncated

        experience = (state, action, reward, next_state, int(failed))
        self.experiences.append(experience)

        return next_state, done

    def train(self, env, n_episodes=3000, n_epochs=40, batch_size=96):
        for episode in safe_tqdm(range(n_episodes), desc="Episode"):
            state, _ = env.reset()
            while True:
                state, done = self.interact_with_env(state, env)

                if len(self.experiences) >= batch_size:
                    experiences = self.process_experiences()
                    for _ in range(n_epochs):
                        self.optimize_model(experiences)

                    self.experiences.clear()

                if done:
                    gc.collect()
                    break

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, fails = experiences
        max_q = self.policy_model(next_states).detach().max(1)[0].unsqueeze(1)

        q_values = self.policy_model(states).gather(1, actions)

        td_targets = rewards + self.gamma * max_q * (1 - fails)

        td_error = td_targets - q_values

        # MSE error
        loss = td_error.pow(2).mul(0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process_experiences(self):
        states = torch.tensor(
            np.stack([exp[0] for exp in self.experiences]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.array([exp[1] for exp in self.experiences]),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.tensor(
            np.array([exp[2] for exp in self.experiences]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.stack([exp[3] for exp in self.experiences]),
            dtype=torch.float32,
            device=self.device,
        )
        fails = torch.tensor(
            np.array([exp[4] for exp in self.experiences]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, fails

    def __call__(self, state):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_model(state_tensor)
            action = q_values.argmax().item()

        return action
