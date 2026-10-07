"""
A minimal DQN agent, with an actual training loop (replay buffer +
Bellman target + gradient update) - not just the action-selection path.

NOTE: `ToyTradingEnv` below is a deliberately simple synthetic
environment (a random walk price series; actions are hold/buy/sell)
so the training loop can run end-to-end with zero external data or
network access. Swap it for a real market-data environment (e.g. a
gymnasium env backed by historical prices) before trusting this for
anything real - and note that DQN on raw trading signals like this is
a teaching example, not a trading strategy.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class ToyTradingEnv:
    """A tiny synthetic single-asset environment: state is a short window
    of past (normalized) price deltas; actions are 0=hold, 1=buy, 2=sell;
    reward is the signed next-step return gated on the chosen action."""

    def __init__(self, window=4, episode_len=200):
        self.window = window
        self.episode_len = episode_len

    def reset(self):
        self.prices = [100.0]
        for _ in range(self.window + self.episode_len):
            self.prices.append(self.prices[-1] * (1 + np.random.randn() * 0.01))
        self.t = self.window
        self.position = 0
        return self._state()

    def _state(self):
        recent = self.prices[self.t - self.window:self.t]
        deltas = np.diff(recent) / recent[:-1]
        return np.append(deltas, self.position).astype(np.float32)

    def step(self, action):
        price_now, price_next = self.prices[self.t], self.prices[self.t + 1]
        step_return = (price_next - price_now) / price_now

        if action == 1:    # buy
            self.position = 1
        elif action == 2:  # sell
            self.position = -1
        reward = self.position * step_return

        self.t += 1
        done = self.t >= self.window + self.episode_len - 1
        return self._state(), reward, done


class DQNAgent:
    def __init__(self, state_dim=4, action_dim=3, gamma=0.99, lr=1e-3):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = ReplayBuffer()

    def select_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.randint(3)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values, dim=1).item()

    def update(self, batch_size=32):
        """One gradient step on a sampled batch - the actual DQN learning
        update (Bellman target via the target network), not just inference."""
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


def train(episodes=50, batch_size=32, target_sync_every=10):
    env = ToyTradingEnv()
    agent = DQNAgent()

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward, losses = 0.0, []
        epsilon = max(0.05, 1.0 - ep / episodes)

        while not done:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update(batch_size)
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward

        if ep % target_sync_every == 0:
            agent.sync_target()

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        print(f"Episode {ep + 1}/{episodes} - reward: {total_reward:.3f} - avg loss: {avg_loss:.4f} - epsilon: {epsilon:.2f}")

    return agent


if __name__ == "__main__":
    train()
