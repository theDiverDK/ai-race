"""Incremental clipped PPO updates for the race's continuous controls."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from neural import Network


class ActorCritic(nn.Module):
    def __init__(self, sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.actor_layers = nn.ModuleList(
            nn.Linear(inputs, outputs) for inputs, outputs in zip(sizes[:-2], sizes[1:-1])
        )
        self.actor_head = nn.Linear(sizes[-2], sizes[-1])
        self.shortcut = nn.Linear(sizes[0], sizes[-1], bias=False)
        critic_layers: list[nn.Module] = []
        previous = sizes[0]
        for width in sizes[1:-1]:
            critic_layers.extend((nn.Linear(previous, width), nn.Tanh()))
            previous = width
        critic_layers.append(nn.Linear(previous, 1))
        self.critic = nn.Sequential(*critic_layers)
        self.log_std = nn.Parameter(torch.full((sizes[-1],), -0.8))
        with torch.no_grad():
            self.actor_head.bias[1] = 0.35
            self.actor_head.bias[2] = -0.6

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = observations
        for layer in self.actor_layers:
            hidden = torch.tanh(layer(hidden))
        mean = self.actor_head(hidden) + self.shortcut(observations)
        return mean, self.critic(observations).squeeze(-1)


class PPOAgent:
    """Collects on-policy batches and trains one minibatch per UI update."""

    def __init__(self, sizes: tuple[int, ...]) -> None:
        torch.set_num_threads(1)
        self.sizes = sizes
        self.model = ActorCritic(sizes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.updates = 0
        self._data: dict[str, torch.Tensor] | None = None
        self._order: torch.Tensor | None = None
        self._cursor = 0
        self._epoch = 0
        self.epochs = 4
        self.minibatch_size = 256
        self.policy_loss = 0.0
        self.value_loss = 0.0

    def act(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            mean, values = self.model(torch.as_tensor(observations, dtype=torch.float32))
            deviation = self.model.log_std.clamp(-2.5, 0.5).exp()
            distribution = Normal(mean, deviation)
            raw = distribution.sample()
            log_probability = distribution.log_prob(raw).sum(dim=-1)
            return (
                torch.tanh(raw).numpy(), raw.numpy(),
                log_probability.numpy(), values.numpy(),
            )

    def value(self, observations: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            _, values = self.model(torch.as_tensor(observations, dtype=torch.float32))
            return values.numpy()

    def export_network(self) -> Network:
        """Mirror the actor in the existing format for the live network inspector."""
        layers = [*self.model.actor_layers, self.model.actor_head]
        return Network(
            self.sizes,
            [layer.weight.detach().tolist() for layer in layers],
            [layer.bias.detach().tolist() for layer in layers],
            self.model.shortcut.weight.detach().tolist(),
        )

    def begin_update(self, rollout: list[dict[str, np.ndarray]]) -> None:
        """Use GAE, bootstrapping truncated episodes but not crashed cars."""
        rewards = np.stack([step["rewards"] for step in rollout])
        values = np.stack([step["values"] for step in rollout])
        next_values = np.stack([step["next_values"] for step in rollout])
        dones = np.stack([step["dones"] for step in rollout])
        advantages = np.zeros_like(rewards, dtype=np.float32)
        following = np.zeros(rewards.shape[1], dtype=np.float32)
        gamma, gae_lambda = 0.99, 0.95
        for t in range(len(rollout) - 1, -1, -1):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            following = delta + gamma * gae_lambda * (1.0 - dones[t]) * following
            advantages[t] = following
        returns = advantages + values
        valid = np.stack([
            step["valid"] if "valid" in step else np.ones_like(step["dones"], dtype=bool)
            for step in rollout
        ]).reshape(-1)
        flat_advantages = advantages.reshape(-1)[valid]
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        self._data = {
            "observations": torch.as_tensor(np.concatenate([step["observations"] for step in rollout])[valid]),
            "raw_actions": torch.as_tensor(np.concatenate([step["raw_actions"] for step in rollout])[valid]),
            "old_logp": torch.as_tensor(np.concatenate([step["logp"] for step in rollout])[valid]),
            "advantages": torch.as_tensor(flat_advantages),
            "returns": torch.as_tensor(returns.reshape(-1)[valid]),
        }
        self._order = torch.randperm(len(flat_advantages))
        self._cursor = 0
        self._epoch = 0

    @property
    def optimizing(self) -> bool:
        return self._data is not None

    def train_minibatch(self) -> bool:
        """Take one optimizer step; return true when the PPO update is done."""
        if self._data is None or self._order is None:
            return False
        batch = self._order[self._cursor:self._cursor + self.minibatch_size]
        data = self._data
        mean, values = self.model(data["observations"][batch])
        deviation = self.model.log_std.clamp(-2.5, 0.5).exp()
        distribution = Normal(mean, deviation)
        logp = distribution.log_prob(data["raw_actions"][batch]).sum(dim=-1)
        ratio = (logp - data["old_logp"][batch]).exp()
        advantage = data["advantages"][batch]
        policy_loss = -torch.minimum(
            ratio * advantage,
            ratio.clamp(0.8, 1.2) * advantage,
        ).mean()
        value_loss = (values - data["returns"][batch]).square().mean()
        entropy = distribution.entropy().sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        self.policy_loss = float(policy_loss.detach())
        self.value_loss = float(value_loss.detach())
        self._cursor += len(batch)
        if self._cursor < len(self._order):
            return False
        self._epoch += 1
        if self._epoch < self.epochs:
            self._order = torch.randperm(len(self._order))
            self._cursor = 0
            return False
        self._data = None
        self._order = None
        self.updates += 1
        return True

    def save(self, path: Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "sizes": self.sizes,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, sizes: tuple[int, ...]) -> "PPOAgent | None":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if tuple(payload["sizes"]) != sizes:
                return None
            agent = cls(sizes)
            agent.model.load_state_dict(payload["model"])
            agent.optimizer.load_state_dict(payload["optimizer"])
            agent.updates = int(payload["updates"])
            return agent
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, KeyError, TypeError, RuntimeError):
            return None
