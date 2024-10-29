from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from termcolor import cprint

if __name__ != "__main__":
    from pendulum_env.SinTransform import SinTransform
    from pendulum_env.CosTransform import CosTransform
else:
    from SinTransform import SinTransform
    from CosTransform import CosTransform

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cprint(f"Using device: {device}", 'green' if torch.cuda.is_available() else 'red')

def _step(tensordict, device="cuda"):
    th, thdot = tensordict["th"], tensordict["thdot"]

    g_force = tensordict["params", "g"]
    mass = tensordict["params", "m"]
    length = tensordict["params", "l"]
    dt = tensordict["params", "dt"]
    max_torque = tensordict["params", "max_torque"]
    max_speed = tensordict["params", "max_speed"]
    u = tensordict["action"].squeeze(-1)

    u = u.clamp(-max_torque, max_torque)

    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    new_thdot = thdot + (
        (3 * g_force / (2 * length) * th.sin() + 3.0 / (mass * length ** 2) * u) * dt
    )
    new_thdot = new_thdot.clamp(-max_speed, max_speed)
    new_th = th + new_thdot * dt

    reward = -costs.unsqueeze(-1)
    done = torch.zeros_like(reward, dtype=torch.bool, device=device)
    out = TensorDict(
        {
            "th": new_th,
            "thdot": new_thdot,
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
        device=device,
    )
    return out

def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size=self.batch_size, device=self.device)

    high_th = torch.tensor(DEFAULT_X, device=self.device)
    high_thdot = torch.tensor(DEFAULT_Y, device=self.device)
    low_th = -high_th
    low_thdot = -high_thdot

    th = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_th - low_th)
        + low_th
    )
    thdot = (
        torch.rand(tensordict.shape, generator=self.rng, device=self.device)
        * (high_thdot - low_thdot)
        + low_thdot
    )
    out = TensorDict(
        {
            "th": th,
            "thdot": thdot,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
        device=self.device,
    )
    return out

def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = CompositeSpec(
        th=BoundedTensorSpec(
            low=-torch.pi,
            high=torch.pi,
            shape=(),
            dtype=torch.float32,
        ),
        thdot=BoundedTensorSpec(
            low=-td_params["params", "max_speed"],
            high=td_params["params", "max_speed"],
            shape=(),
            dtype=torch.float32,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = BoundedTensorSpec(
        low=-td_params["params", "max_torque"],
        high=td_params["params", "max_torque"],
        shape=(1,),
        dtype=torch.float32,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _set_seed(self, seed: Optional[int], cuda: bool = True):
    if cuda:
        # if the device is CPU, we can use the torch random number generator
        rng = torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    self.rng = rng

def gen_params(g=10.0, batch_size=None, device='cuda') -> TensorDictBase:
    """Returns a tensordict containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": torch.tensor(8, device=device),
                    "max_torque": torch.tensor(2.0, device=device),
                    "dt": torch.tensor(0.05, device=device),
                    "g": torch.tensor(g, device=device),
                    "m": torch.tensor(1.0, device=device),
                    "l": torch.tensor(1.0, device=device),
                },
                [],
                device=device,
            )
        },
        [],
        device=device,
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class PendulumEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cuda"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed

def simple_rollout(env, steps=100):
    # preallocate:
    data = TensorDict({}, [steps])
    # reset
    _data = env.reset()
    for i in range(steps):
        _data["action"] = env.action_spec.rand()
        _data = env.step(_data)
        data[i] = _data
        _data = step_mdp(_data, keep_other=True)
    return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = PendulumEnv(device="cuda")
    check_env_specs(env)

    td = env.reset()
    # print("reset tensordict", td)

    td = env.rand_step(td)
    # print("random step tensordict", td)

    env = TransformedEnv(
        env,
        # Unsqueeze the observations that we will concatenate
        UnsqueezeTransform(
            dim=-1,
            in_keys=["th", "thdot"],
            in_keys_inv=["th", "thdot"],
        ),
    )

    t_sin = SinTransform(in_keys=["th"], out_keys=["sin"])
    t_cos = CosTransform(in_keys=["th"], out_keys=["cos"])
    env.append_transform(t_sin)
    env.append_transform(t_cos)

    cat_transform = CatTensors(
                    in_keys=["sin", "cos", "thdot"],
                    dim=-1, 
                    out_key="observation", 
                    del_keys=False
                    )
    env.append_transform(cat_transform)

    check_env_specs(env)

    # print("data from rollout:", simple_rollout(env=env, steps=100))

    # batch_size = 10  # number of environments to be executed in batch
    # td = env.reset(env.gen_params(batch_size=[batch_size]))
    # print("reset (batch size of 10)", td)
    # td = env.rand_step(td)
    # print("rand step (batch size of 10)", td)


    # rollout = env.rollout(
    #     3,
    #     auto_reset=False,  # we're executing the reset out of the ``rollout`` call
    #     tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
    # )
    # print("rollout of len 3 (batch size of 10):", rollout)

    torch.manual_seed(0)
    env.set_seed(0)

    hidden_size = 64
    net = nn.Sequential(
        nn.Linear(3, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, 1),
    ).to(device)  # Move network to the correct device


    policy = TensorDictModule(
        net,
        in_keys=["observation"],
        out_keys=["action"],
    )

    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

    batch_size = 512
    episodes = 600_000
    pbar = tqdm.tqdm(range(episodes // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, episodes)
    logs = defaultdict(list)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i_episode in pbar:
        init_td = env.reset(env.gen_params(batch_size=[batch_size], device=device))
        init_td = init_td.to(device)  # Move to the correct device
        rollout = env.rollout(100, policy, tensordict=init_td, auto_reset=False)
        traj_return = rollout["next", "reward"].mean()
        (-traj_return).backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
        scheduler.step()

        # Save the model
        checkpoint = episodes // batch_size // 10
        if i_episode % checkpoint == 0 and i_episode > 0:
            torch.save(policy.state_dict(), f"pendulum_policy_ep{i_episode}.pth")

        # Update the plots
        axes[0].cla()
        axes[0].plot(logs["return"])
        axes[0].set_title("Returns")
        axes[0].set_xlabel("Iteration")

        axes[1].cla()
        axes[1].plot(logs["last_reward"])
        axes[1].set_title("Last Reward")
        axes[1].set_xlabel("Iteration")

        plt.draw()
        plt.pause(0.001)  # Brief pause to update the plot
    plt.savefig(f"results_bs{batch_size}_eps{episodes}.png")
    plt.show()