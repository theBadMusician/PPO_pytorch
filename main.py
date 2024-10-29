import torch
from PPOAgent import PPOAgent
from pendulum_env.PendulumEnv import PendulumEnv
from collections import defaultdict
from tqdm import tqdm

# Configuration
config = {
    "goal_position": 10.0,
    "max_steps": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "frames_per_batch": 1000,
    "total_frames": 50000,
    "sub_batch_size": 64,
    "num_epochs": 10,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "lmbda": 0.95,
    "entropy_eps": 1e-4,
    "lr": 3e-4,
}

# Initialize environment and agent
env = SimpleGoalEnv(goal_position=config["goal_position"], max_steps=config["max_steps"], device=config["device"])
agent = PPOAgent(env, config)

# Optimizer and scheduler
optimizer = torch.optim.Adam(agent.policy_module.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["total_frames"] // config["frames_per_batch"], 0.0)

# Logging setup
logs = defaultdict(list)
pbar = tqdm(total=config["total_frames"])

# Training loop
for batch in agent.collector:
    for _ in range(config["num_epochs"]):
        # Compute advantages and values for the collected batch
        agent.advantage_module(batch)
        
        # Flatten data for sampling
        flat_batch = batch.reshape(-1)
        
        for _ in range(config["frames_per_batch"] // config["sub_batch_size"]):
            # Sample a mini-batch for optimization
            sub_batch = flat_batch.sample(config["sub_batch_size"]).to(config["device"])
            
            # Compute PPO loss
            loss_dict = agent.loss_module(sub_batch)
            loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Update logging
    logs["reward"].append(batch["next", "reward"].mean().item())
    pbar.update(batch.numel())
    
    # Learning rate scheduling
    scheduler.step()
    
    # Display metrics
    pbar.set_description(f"Avg Reward: {logs['reward'][-1]:.4f}")

pbar.close()
print("Training completed.")
