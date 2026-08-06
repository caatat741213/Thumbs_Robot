import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Any
from .actor_critic_network import ActorCriticNetwork
from .rollout_buffer import RolloutBuffer

class PPOAgent:
    """
    PPO Agent class managing the continuous Actor-Critic network, 
    selecting actions and executing Clipped Policy updates.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
        initial_log_std: float = -0.5,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False, 
        return_dist_params: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Choose action and compute log probability and value during rollout collection.
        """
        state_t = state.to(self.device).float()
        with torch.no_grad():
            dist, value = self.network(state_t)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(axis=-1)
            
        if return_dist_params:
            return action.cpu(), log_prob.cpu(), value.cpu(), dist.mean.cpu(), dist.stddev.cpu()
        return action.cpu(), log_prob.cpu(), value.cpu()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform multiple update epochs over rollout data.
        Updates Policy Network (Actor) and Value Network (Critic).
        """
        import numpy as np
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        total_loss_epoch = 0.0
        clip_fraction_epoch = 0.0
        grad_norm_epoch = 0.0
        approx_kl_epoch = 0.0
        num_updates = 0

        # Perform PPO epochs updates
        for epoch in range(self.ppo_epochs):
            generator = buffer.get_generator(self.batch_size)
            
            for batch in generator:
                states = batch["states"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                old_values = batch["old_values"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Mini-batch level advantage normalization for numeric stability
                if advantages.numel() > 1:
                    adv_std = advantages.std()
                    if torch.isfinite(adv_std) and adv_std > 1e-6:
                        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

                # Evaluate new action distribution and state values
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)
                
                # Flatten arrays to exactly 1D to prevent broadcast mismatches
                new_values = new_values.view(-1)
                returns = returns.view(-1)

                # 1. Policy Ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Calculate approximate KL divergence for diagnostics
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()

                # 2. Clipped Actor Loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Calculate clip fraction
                with torch.no_grad():
                    clipped = (ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)
                    clip_fraction = clipped.float().mean().item()

                # 3. Critic Loss (MSE)
                critic_loss = 0.5 * nn.functional.mse_loss(new_values, returns)

                # 4. Entropy regularizer (exploration bonus)
                entropy_loss = -entropy.mean()

                # Total Loss computation
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += actor_loss.item()
                value_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                total_loss_epoch += loss.item()
                clip_fraction_epoch += clip_fraction
                grad_norm_epoch += grad_norm.item()
                approx_kl_epoch += approx_kl
                num_updates += 1

        # Calculate explained variance of values predictions post-updates
        with torch.no_grad():
            states_all = buffer.states
            returns_all = buffer.returns.view(-1)
            _, _, _, values_all = self.network.get_action_and_value(states_all)
            values_all = values_all.view(-1)
            
            y_pred = values_all.cpu().numpy()
            y_true = returns_all.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 0.0 if var_y < 1e-8 else 1.0 - np.var(y_true - y_pred) / var_y

        # Return average metrics for mapping and logs
        return {
            "actor_loss": policy_loss_epoch / num_updates,
            "critic_loss": value_loss_epoch / num_updates,
            "entropy": -entropy_loss_epoch / num_updates,
            "total_loss": total_loss_epoch / num_updates,
            "clip_fraction": clip_fraction_epoch / num_updates,
            "grad_norm": grad_norm_epoch / num_updates,
            "explained_variance": explained_var,
            "approx_kl": approx_kl_epoch / num_updates,
            "lr": self.optimizer.param_groups[0]['lr']
        }

    def save_checkpoint(self, path: str):
        """
        Save network and optimizer states.
        """
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load network checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
