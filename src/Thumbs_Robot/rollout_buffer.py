import torch
from typing import Dict, Tuple, List, Generator

class RolloutBuffer:
    """
    On-policy storage buffer for storing trajectory transitions.
    Computes Generalized Advantage Estimation (GAE) and PPO training batches.
    """
    def __init__(self, size: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.size = size
        self.device = device

        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)  # Represents terminated
        self.truncates = torch.zeros(size, dtype=torch.float32, device=device)  # Represents truncated
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)

        # Output targets computed at rollout end
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        
        self.ptr = 0

    def insert(self, state: torch.Tensor, action: torch.Tensor, reward: float, done: bool, truncated: bool, log_prob: torch.Tensor, value: torch.Tensor):
        """
        Insert a transition step into the buffer.
        """
        if self.ptr >= self.size:
            raise IndexError("RolloutBuffer is full, call compute_advantages and clear before insertion.")
        
        self.states[self.ptr] = state.detach()
        self.actions[self.ptr] = action.detach()
        self.rewards[self.ptr] = torch.tensor(reward, device=self.device)
        self.dones[self.ptr] = torch.tensor(float(done), device=self.device)
        self.truncates[self.ptr] = torch.tensor(float(truncated), device=self.device)
        self.log_probs[self.ptr] = log_prob.detach()
        self.values[self.ptr] = value.detach().squeeze()

        self.ptr += 1

    def clear(self):
        """
        Reset pointer to zero, clearing the buffer.
        """
        self.ptr = 0

    def compute_advantages(self, next_value: torch.Tensor, next_done: bool, next_truncated: bool, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Calculate GAE advantages:
        delta_t = r_t + gamma * (1 - terminated_t+1) * V(s_t+1) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - ended_t+1) * A_t+1
        """
        last_gae = 0.0
        next_value = next_value.squeeze().item()
        next_done = float(next_done) # Representing terminated
        next_truncated = float(next_truncated) # Representing truncated

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_val = next_value
                # If the episode ends due to termination (success), we mask future reward (non_terminal = 0.0).
                # If it ends due to truncation, or doesn't end, we bootstrap the future reward (non_terminal = 1.0).
                non_terminal = 1.0 - next_done
                # Propagation should stop if the episode ended by either termination or truncation.
                episode_continue = 1.0 - float(next_done or next_truncated)
            else:
                next_val = self.values[t + 1].item()
                non_terminal = 1.0 - self.dones[t + 1].item()
                episode_continue = 1.0 - float(self.dones[t + 1].item() or self.truncates[t + 1].item())

            delta = self.rewards[t].item() + gamma * non_terminal * next_val - self.values[t].item()
            last_gae = delta + gamma * gae_lambda * episode_continue * last_gae
            self.advantages[t] = last_gae

        # TD Targets (Returns) = Advantage + Value
        self.returns = self.advantages + self.values

        # Rollout level standardization
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_generator(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generator yielding mini-batches for PPO updates.
        """
        indices = torch.randperm(self.size, device=self.device)
        for start in range(0, self.size, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "states": self.states[batch_idx],
                "actions": self.actions[batch_idx],
                "old_log_probs": self.log_probs[batch_idx],
                "old_values": self.values[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
            }
