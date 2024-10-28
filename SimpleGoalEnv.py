import torch
from torchrl.data.tensor_specs import Bounded, Composite
from torchrl.envs import EnvBase
from tensordict import TensorDict
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt


import torch
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import Bounded, Composite
from tensordict import TensorDict

class SimpleGoalEnv(EnvBase):
    def __init__(self, goal_position=10.0, max_steps=100, device="cuda"):
        super().__init__(device=torch.device(device))
        
        # Ensure goal_position has shape [1] for consistent operations
        self.goal_position = torch.tensor([goal_position], device=self.device)
        self.max_steps = max_steps
        self.current_step = 0

        # Define observation and action dimensions
        self.obs_dim = 1
        self.action_dim = 1

        # Define explicit low and high tensors for action clamping
        self.action_low = torch.tensor([-1.0], device=self.device)
        self.action_high = torch.tensor([1.0], device=self.device)

        # Define observation and action specs using Composite
        self.observation_spec = Composite(
            observation=Bounded(
                low=torch.tensor([-20.0], device=self.device),
                high=torch.tensor([20.0], device=self.device),
                shape=(self.obs_dim,),
                dtype=torch.float32,
                device=self.device,
            )
        )
        
        self.action_spec = Composite(
            action=Bounded(
                low=self.action_low,
                high=self.action_high,
                shape=(self.action_dim,),
                dtype=torch.float32,
                device=self.device,
            )
        )

        # Initialize position, velocity, and acceleration as tensors with shape [1]
        self.position = torch.zeros(1, device=self.device)
        self.velocity = torch.zeros(1, device=self.device)
        self.acceleration = torch.zeros(1, device=self.device)

    def _reset(self, tensordict=None):
        """Resets the environment and returns the initial observation as a TensorDict."""
        self.position = torch.zeros_like(self.position)
        self.velocity = torch.zeros_like(self.velocity)
        self.acceleration = torch.zeros_like(self.acceleration)
        self.current_step = 0

        # Return the initial observation with correct shape
        return TensorDict(
            {"observation": self.position.clone()},
            batch_size=[],
            device=self.device
        )

    def _step(self, tensordict):
        """Performs an action and updates the environment state."""
        action = tensordict["action"].to(self.device)
        action = torch.clamp(action, min=self.action_low, max=self.action_high)

        # Force, slope, and friction calculations
        mass = 1.0
        self.acceleration = action / mass

        hill_slope = torch.tanh(self.position / self.goal_position)
        gravitational_force = -9.8 * hill_slope
        friction = -0.1 * self.velocity
        self.acceleration += (gravitational_force + friction) / mass

        # Update velocity and position, ensuring shape consistency
        self.velocity = (self.velocity + self.acceleration)
        self.position = (self.position + self.velocity + 0.5 * self.acceleration)

        # Update step count
        self.current_step += 1

        # Calculate distance to goal and enforce scalar output
        distance_to_goal = torch.abs(self.goal_position - self.position)
        if distance_to_goal.numel() > 1:
            distance_to_goal = distance_to_goal.mean()
        reward = -distance_to_goal.item()

        done = self.current_step >= self.max_steps or distance_to_goal < 0.5

        # Return the updated state as a TensorDict with correct shape for observation
        return TensorDict(
            {
                "observation": self.position.clone(),
                "reward": torch.tensor(reward, device=self.device),
                "done": torch.tensor(done, device=self.device),
            },
            batch_size=[],
            device=self.device
        )

    def _set_seed(self, seed):
        """Sets the random seed for reproducibility."""
        torch.manual_seed(seed)
        return seed


if __name__ == "__main__":
    # Example usage with CUDA
    env = SimpleGoalEnv(device="cuda")

    # Random policy function for testing
    def random_policy(observation):
        return torch.rand(1, device=observation.device) * 2 - 1  # Random action within the bounds of [-1, 1]

    # Rayleigh policy function for testing
    def rayleigh_policy(observation):
        rayleigh = np.random.rayleigh(1.0) * 2 - 1  # Rayleigh distribution with scale=1
        return torch.tensor([rayleigh], device=observation.device)

    # Test rollout
    observations, rewards = env.rollout(steps=100, policy=rayleigh_policy)
    print("Observations:", observations)
    print("Rewards:", rewards)

    plt.figure(figsize=(8, 6))
    plt.plot(observations.cpu().numpy(), label="Position")
    plt.axhline(env.goal_position.item(), color="r", linestyle="--", label="Goal Position")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Agent's Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()
