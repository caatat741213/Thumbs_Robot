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
        # 1. Compute Actor policy mean and std
        x = self.actor_backbone(state)
        mu = torch.tanh(self.mu_head(x)) # Bound mean between [-1, 1]
        
        std = torch.exp(self.log_std.clamp(-20, 2)) # Numeric stability clipping
        dist = Normal(mu, std)

        # 2. Compute Critic value
        value = self.critic(state)

        return dist, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Retrieve only state value V(s) for advantage calculation.
        """
        return self.critic(state)

    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action actions, log probabilities, entropy and state value.
        Used primarily during PPO updates.
        """
        dist, value = self.forward(state)
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return action, log_prob, entropy, value
