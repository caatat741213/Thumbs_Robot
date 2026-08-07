"""
G1 Hand-Gesture Actor-Critic Network (PPO)
Stage: Phase 2 (PPO Network Design)
Primary Function:
- Defines the Multilayer Perceptron (MLP) mapping state observations (dim=32) to continuous Gaussian actions (dim=5).
- Houses separate network backbones for Actor policy and Critic state-value estimations.
- Calculates log probabilities, entropy values, and predictions for PPO updating updates.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

class ActorCriticNetwork(nn.Module):
    """
    Continuous Actor-Critic Network (MLP) for G1 continuous motor control.
    Contains separate Actor and Critic heads.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, initial_log_std: float = -0.5):
        super(ActorCriticNetwork, self).__init__()

        # Actor MLP: predicts mean (mu) of action distribution
        self.actor_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        
        # Log standard deviation parameter (std = exp(log_std))
        # Initialized with log_std (default -0.5 -> std = ~0.6)
        self.log_std = nn.Parameter(torch.ones(action_dim) * initial_log_std)

        # Critic MLP: predicts state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - dist (Normal): Gaussian action distribution
            - value (Tensor): estimated state value V(s)
        """
        # 1. Compute Actor policy mean (mu) and standard deviation (std)
        x = self.actor_backbone(state)
        mu = torch.tanh(self.mu_head(x)) # Bound mean between [-1, 1] for hard-limit clipping consistency
        
        # Clip log_std to enforce numerical stability bounds
        std = torch.exp(self.log_std.clamp(-20, 2))
        dist = Normal(mu, std)

        # 2. Compute Critic state value V(s)
        value = self.critic(state)

        return dist, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Retrieve only state value V(s) for advantage calculation bootstrapping.
        """
        return self.critic(state)

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action outputs, log probabilities, entropy and state value.
        Used primarily during policy training epochs.
        """
        dist, value = self.forward(state)
        if action is None:
            action = dist.sample()
        
        # Calculate sum of log probabilities and distributions entropy across action dimensions
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return action, log_prob, entropy, value
