from typing import Dict

import torch
import torch.nn as nn


class MLPBlock(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalMemoryEncoder(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_embed_dim: int = 64,
        token_dim: int = 64,
        memory_dim: int = 64,
        hidden_dim: int = 128,
        aggregator_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.token_dim = int(token_dim)
        self.memory_dim = int(memory_dim)

        gru_dropout = float(dropout) if int(aggregator_layers) > 1 else 0.0

        self.action_encoder = MLPBlock(
            input_dim=self.action_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_embed_dim,
            dropout=dropout,
        )
        self.transition_norm = nn.LayerNorm(self.obs_dim + self.action_embed_dim)
        self.transition_encoder = MLPBlock(
            input_dim=self.obs_dim + self.action_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=self.token_dim,
            dropout=dropout,
        )
        self.aggregator = nn.GRU(
            input_size=self.token_dim,
            hidden_size=self.memory_dim,
            num_layers=int(aggregator_layers),
            dropout=gru_dropout,
            batch_first=True,
        )

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps = actions.shape[:2]
        action_embed = self.action_encoder(actions.reshape(batch_size * time_steps, -1))
        return action_embed.reshape(batch_size, time_steps, -1)

    def forward(
        self,
        obs_latents: torch.Tensor,
        aligned_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if obs_latents.ndim != 3:
            raise ValueError(f"obs_latents must be [B, T, D], got {tuple(obs_latents.shape)}")
        if aligned_actions.ndim != 3:
            raise ValueError(f"aligned_actions must be [B, T-1, A], got {tuple(aligned_actions.shape)}")
        if obs_latents.shape[1] != aligned_actions.shape[1] + 1:
            raise ValueError(
                f"Expected obs_latents length = aligned_actions length + 1, got "
                f"{obs_latents.shape[1]} and {aligned_actions.shape[1]}"
            )

        z_curr = obs_latents[:, :-1, :]
        z_next = obs_latents[:, 1:, :]
        action_embed = self._encode_actions(aligned_actions)
        transition_pairs = torch.cat([z_curr, action_embed], dim=-1)
        transition_input = self.transition_norm(transition_pairs)

        batch_size, time_steps = transition_input.shape[:2]
        causal_tokens = self.transition_encoder(transition_input.reshape(batch_size * time_steps, -1))
        causal_tokens = causal_tokens.reshape(batch_size, time_steps, -1)

        memory_states, hidden = self.aggregator(causal_tokens)
        memory = hidden[-1]
        return {
            "memory": memory,
            "memory_states": memory_states,
            "causal_tokens": causal_tokens,
            "aligned_actions": aligned_actions,
            "action_embed": action_embed,
            "z_curr": z_curr,
            "z_next": z_next,
            "transition_pairs": transition_pairs,
        }
