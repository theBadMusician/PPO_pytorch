import torch
from torch import nn
from torch.optim import Adam
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

class PPOAgent:
    def __init__(self, env, config):
        self.env = env
        self.device = config.get("device", torch.device("cpu"))
        self.num_cells = config.get("num_cells", 256)
        self.lr = config.get("lr", 3e-4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        
        self.frames_per_batch = config.get("frames_per_batch", 1000)
        self.total_frames = config.get("total_frames", 50000)
        self.sub_batch_size = config.get("sub_batch_size", 64)
        self.num_epochs = config.get("num_epochs", 10)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.lmbda = config.get("lmbda", 0.95)
        self.entropy_eps = config.get("entropy_eps", 1e-4)

        # Model setup
        self.policy_module = self._build_policy()
        self.value_module = self._build_value()
        
        # Loss setup
        self.advantage_module = GAE(gamma=self.gamma, lmbda=self.lmbda, value_network=self.value_module)
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            critic_coef=1.0,
            loss_critic_type="smooth_l1"
        )
        
        # Optimizer
        self.optim = Adam(self.loss_module.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.total_frames // self.frames_per_batch, 0.0)
        
        # Data collector and replay buffer
        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device
        )
        self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=self.frames_per_batch), sampler=SamplerWithoutReplacement())

    def _build_policy(self):
        actor_net = nn.Sequential(
            nn.Linear(self.env.observation_spec.shape[0], self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, 2 * self.env.action_spec.shape[-1]),
        )
        return actor_net.to(self.device)

    def _build_value(self):
        value_net = nn.Sequential(
            nn.Linear(self.env.observation_spec.shape[0], self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, self.num_cells),
            nn.Tanh(),
            nn.Linear(self.num_cells, 1),
        )
        return value_net.to(self.device)

    def train(self):
        logs = defaultdict(list)
        pbar = tqdm(total=self.total_frames)

        for i, tensordict_data in enumerate(self.collector):
            for _ in range(self.num_epochs):
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(self.frames_per_batch // self.sub_batch_size):
                    subdata = self.replay_buffer.sample(self.sub_batch_size).to(self.device)
                    loss_vals = self.loss_module(subdata)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            if i % 10 == 0:
                self.evaluate(logs)
            self.scheduler.step()

        pbar.close()
        self.plot_results(logs)

    def evaluate(self, logs):
        with torch.no_grad():
            eval_rollout = self.env.rollout(1000, self.policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())

    def plot_results(self, logs):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("Training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()
