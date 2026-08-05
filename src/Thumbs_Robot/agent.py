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

        self.network = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return action.cpu(), log_prob.cpu(), value.cpu()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Perform multiple update epochs over rollout data.
        Updates Policy Network (Actor) and Value Network (Critic).
        """
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        total_loss_epoch = 0.0
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

                # Evaluate new action distribution and state values
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)
                new_values = new_values.squeeze()

                # 1. Policy Ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # 2. Clipped Actor Loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 3. Critic Loss (MSE)
                critic_loss = 0.5 * nn.functional.mse_loss(new_values, returns)

                # 4. Entropy regularizer (exploration bonus)
                entropy_loss = -entropy.mean()

                # Total Loss computation
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # Gradient descent step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += actor_loss.item()
                value_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                total_loss_epoch += loss.item()
                num_updates += 1

        # Return average metrics for mapping and logs
        return {
            "actor_loss": policy_loss_epoch / num_updates,
            "critic_loss": value_loss_epoch / num_updates,
            "entropy": -entropy_loss_epoch / num_updates,
            "total_loss": total_loss_epoch / num_updates
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
