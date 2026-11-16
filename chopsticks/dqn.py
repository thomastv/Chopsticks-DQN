"""Action-Masked Double Dueling DQN with Zero-Sum Negamax Bellman updates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chopsticks.env import NUM_ACTIONS, STATE_DIM, State, encode_state, get_legal_action_mask

# Avoid OpenMP thread-pool contention and Windows process-teardown hangs on small MLPs
torch.set_num_threads(1)


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network with legal-action-masked advantage centering."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(
        self,
        state_vec: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Q(s, a) with optional legal-action-masked advantage normalization.

        Args:
            state_vec: Float tensor of shape (B, STATE_DIM) or (STATE_DIM,).
            legal_mask: Optional boolean tensor of shape (B, NUM_ACTIONS) or (NUM_ACTIONS,).
        """
        single = state_vec.dim() == 1
        if single:
            state_vec = state_vec.unsqueeze(0)
            if legal_mask is not None:
                legal_mask = legal_mask.unsqueeze(0)

        features = self.trunk(state_vec)
        value = self.value_stream(features)  # (B, 1)
        advantage = self.advantage_stream(features)  # (B, NUM_ACTIONS)

        if legal_mask is not None:
            mask_f = legal_mask.to(dtype=advantage.dtype)
            legal_counts = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean_adv = (advantage * mask_f).sum(dim=-1, keepdim=True) / legal_counts
        else:
            mean_adv = advantage.mean(dim=-1, keepdim=True)

        q_vals = value + (advantage - mean_adv)
        return q_vals.squeeze(0) if single else q_vals


class ReplayBuffer:
    """Circular experience replay buffer for turn-relative zero-sum transitions."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.legal_masks = np.zeros((capacity, NUM_ACTIONS), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.next_legal_masks = np.zeros((capacity, NUM_ACTIONS), dtype=bool)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(
        self,
        state_vec: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        reward: float,
        next_state_vec: np.ndarray,
        next_legal_mask: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.states[idx] = state_vec
        self.legal_masks[idx] = legal_mask
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state_vec
        self.next_legal_masks[idx] = next_legal_mask
        self.dones[idx] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.legal_masks[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_states[indices]).to(device),
            torch.from_numpy(self.next_legal_masks[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """Action-Masked Double Dueling DQN Agent."""

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        hidden_dim: int = 128,
        buffer_capacity: int = 50_000,
        device: str | torch.device | None = None,
        seed: int = 42,
    ) -> None:
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.tau = tau

        self.online_net = DuelingDQN(hidden_dim=hidden_dim).to(self.device)
        self.target_net = DuelingDQN(hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(self.online_net.parameters(), lr=lr, weight_decay=1e-5)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    @torch.no_grad()
    def get_q_values(self, state: State, mask_illegal: bool = True) -> np.ndarray:
        """Compute Q(s, a) for all 5 actions in `state`."""
        state_vec = torch.from_numpy(encode_state(state)).to(self.device)
        legal_mask_np = get_legal_action_mask(state)
        legal_mask_t = torch.from_numpy(legal_mask_np).to(self.device)

        self.online_net.eval()
        q_vals = self.online_net(state_vec, legal_mask_t).cpu().numpy()
        if mask_illegal:
            q_vals = np.where(legal_mask_np, q_vals, -np.inf)
        return q_vals

    def select_action(
        self,
        state: State,
        legal_mask: np.ndarray | None = None,
        epsilon: float = 0.0,
    ) -> int:
        """Select a legal action using action-masked epsilon-greedy exploration."""
        if legal_mask is None:
            legal_mask = get_legal_action_mask(state)
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise ValueError(f"No legal actions available in state {state}")

        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(legal_indices))

        q_vals = self.get_q_values(state, mask_illegal=True)
        # Break exact ties randomly among legal actions
        max_q = np.max(q_vals[legal_indices])
        best_actions = legal_indices[np.isclose(q_vals[legal_indices], max_q, atol=1e-6)]
        return int(self.rng.choice(best_actions))

    def update(self, batch_size: int = 128) -> float | None:
        """Perform one Double DQN Negamax Bellman update step."""
        if len(self.buffer) < batch_size:
            return None

        (
            states,
            legal_masks,
            actions,
            rewards,
            next_states,
            next_legal_masks,
            dones,
        ) = self.buffer.sample(batch_size, self.rng, self.device)

        self.online_net.train()
        q_all = self.online_net(states, legal_masks)
        q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select greedy next action with online_net, evaluate with target_net
            next_q_online = self.online_net(next_states, next_legal_masks)
            masked_next_q_online = next_q_online.masked_fill(~next_legal_masks, float("-inf"))
            best_next_actions = masked_next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states, next_legal_masks)
            next_v = next_q_target.gather(1, best_next_actions).squeeze(1)
            # Zero out next_v where episode is done (or no legal actions exist)
            has_legal = next_legal_masks.any(dim=1)
            valid_next = (1.0 - dones) * has_legal.to(dtype=dones.dtype)

            # Zero-sum Negamax backup: opponent's gain is current player's loss
            targets = rewards - self.gamma * valid_next * next_v

        loss = F.smooth_l1_loss(q_pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for param, target_param in zip(
                self.online_net.parameters(), self.target_net.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

        return float(loss.item())

    def save(self, path: str | Path) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "gamma": self.gamma,
            },
            ckpt_path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.online_net.eval()
        self.target_net.eval()
