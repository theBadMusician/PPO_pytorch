import torch
from torchrl.data.tensor_specs import Bounded, Composite
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt
import scipy.stats as stats

class SimpleGoalEnv:
    def __init__(self, goal_position=10.0, max_steps=100, device="cuda"):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            device = "cpu"
        cprint("Using device: {}".format(device), "yellow" if device == "cpu" else "green")
        self.device = torch.device(device)

        self.goal_position = torch.tensor(goal_position, device=self.device)  # Ensure goal is on the correct device
        self.max_steps = max_steps
        self.current_step = 0

        # Define observation spec: agent's current position (bounded between -20 and 20)
        self.observation_spec = Bounded(
            low=torch.tensor([-20.0], device=self.device),
            high=torch.tensor([20.0], device=self.device),
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )

        # Define action spec: single movement direction and magnitude, bounded between -1 and 1
        self.action_spec = Bounded(
            low=torch.tensor([-10.0], device=self.device),
            high=torch.tensor([10.0], device=self.device),
            shape=(1,),
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize agent position on the specified device
        self.position = torch.zeros(1, device=self.device)
        self.velocity = torch.zeros(1, device=self.device)
        self.acceleration = torch.zeros(1, device=self.device)

    def reset(self):
        """Resets the environment and returns the initial observation."""
        self.position = torch.zeros(1, device=self.device)  # Start at position 0
        self.current_step = 0
        return self.position.clone()

    def step(self, action):
        """Performs an action and updates the environment state.

        Args:
            action (torch.Tensor): 1D tensor with one element representing the movement direction.

        Returns:
            observation (torch.Tensor): Current position of the agent.
            reward (float): Reward obtained for the current step.
            done (bool): Whether the episode has ended.
            info (dict): Additional information (empty here).
        """
        # Ensure action is on the correct device and within bounds
        action = action.to(self.device)
        action = torch.clamp(action, min=self.action_spec.space.low, max=self.action_spec.space.high)

        # Mass of the agent
        mass = 1.0

        # Force applied by the action, resulting in acceleration
        self.acceleration = action / mass

        # Calculate the slope of the hill at the current position
        # Example: Assume a simple hill gradient function (e.g., steeper as the agent approaches the goal)
        # hill_slope = torch.tanh(self.position / self.goal_position)  # Arbitrary slope function

        possible_positions = torch.arange(-20, 20, 0.1).to(self.device)

        # hill slopes looks like gaussian distribution
        mu = 0
        variance = 1
        sigma = np.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, len(possible_positions.cpu().numpy()))
        hill_slopes = stats.norm.pdf(x, mu, sigma)
        hill_slopes = torch.tensor(hill_slopes, device=self.device)
        hill_slope = torch.tensor(hill_slopes[torch.argmin(torch.abs(possible_positions - self.position))], device=self.device)
        # plt.figure(figsize=(8, 6))
        # plt.plot(possible_positions.cpu().numpy(), hill_slopes, label="Hill Slope")
        # plt.axvline(0, color="r", linestyle="--", label="Goal Position")
        # plt.xlabel("Position")
        # plt.ylabel("Hill Slope")
        # plt.title("Hill Slope Function")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        gravitational_force = -9.8 * hill_slope  # Gravitational component along the hill

        # Frictional force, opposing motion, proportional to velocity
        friction = -0.1 * self.velocity

        # Update acceleration with gravitational and frictional forces
        self.acceleration += (gravitational_force + friction) / mass

        # Update velocity: v = u + at
        self.velocity += self.acceleration

        # Update position: s = ut + 0.5 * a * t^2
        self.position += self.velocity + 0.5 * self.acceleration

        # Update step count
        self.current_step += 1

        # Reward: Negative of the absolute distance to the goal position
        distance_to_goal = torch.abs(self.goal_position - self.position)
        reward = -distance_to_goal.item()

        # Episode termination conditions
        done = self.current_step >= self.max_steps or distance_to_goal < 0.5

        return self.position.clone(), reward, done, {}

    def rollout(self, steps, policy):
        """Performs a rollout using the given policy for evaluation purposes.

        Args:
            steps (int): Number of steps to simulate.
            policy (callable): Policy function that takes an observation and returns an action.

        Returns:
            torch.Tensor: Tensor of observations collected during the rollout.
            torch.Tensor: Tensor of rewards collected during the rollout.
        """
        observations, rewards = [], []
        observation = self.reset()
        for _ in range(steps):
            action = policy(observation)
            observation, reward, done, _ = self.step(action)
            observations.append(observation.clone())
            rewards.append(reward)
            if done:
                break
        return torch.stack(observations), torch.tensor(rewards, device=self.device)


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
