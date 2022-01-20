from comet_ml import Experiment

# This block is so that comet_ml is always imported first
if True:
    pass

import random
from typing import Any

import gin
import gym
import torch as t
from days.atari_wrappers import AtariWrapper
from torch import nn
from einops import rearrange

import dqn

EXPERIMENT = Experiment(
    api_key="qjxcybqq2HsGHbEwATgNiqWgE",
    project_name="mlab_atari_v1",
    workspace="ttwang",
    auto_metric_logging=False,
)


class AtariConv(nn.Module):
    def __init__(self, obs_n_channels, n_action_space):
        super().__init__()
        self.obs_n_channels = obs_n_channels
        self.n_action_space = n_action_space
        self.layers = nn.Sequential(
            nn.Conv2d(obs_n_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, n_action_space),
        )

    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = rearrange(obs, "h w c -> 1 c h w")
            return self.layers(obs)[0]

        assert len(obs.shape) == 4
        obs = rearrange(obs, "b h w c -> b c h w")
        return self.layers(obs)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@gin.configurable
def train(
    seed: int,
    device: str,
    n_train_steps: int,
    max_buffer_size: int,
    train_freq: int,
    eval_freq: int,
    video_freq: int,
    batch_size: int,
    greedy_eps_start: float,
    greedy_eps_end: float,
    discount_factor: float,
    lr: float,
):
    t.manual_seed(seed)
    random.seed(seed)
    ENV_NAME = "BreakoutNoFrameskip-v0"

    env_atari = gym.make(ENV_NAME)
    env_atari = AtariWrapper(env_atari)
    model_atari = AtariConv(
        obs_n_channels=env_atari.observation_space.shape[-1],
        n_action_space=env_atari.action_space.n,
    ).to(device)

    dqn.dqn_train(
        env=env_atari,
        model=model_atari,
        n_steps=n_train_steps,
        max_buffer_size=max_buffer_size,
        train_freq=train_freq,
        eval_freq=eval_freq,
        video_freq=video_freq,
        batch_size=batch_size,
        eps_start=greedy_eps_start,
        eps_end=greedy_eps_end,
        gamma=discount_factor,
        lr=lr,
        seed=seed,
        experiment=EXPERIMENT,
    )

    t.save(model_atari.state_dict(), "models/dqn_atari.pt")
    EXPERIMENT.log_asset("models/dqn_atari.pt")


def flatten_gin_config(
    d: dict[tuple[str, str], tuple[str, Any]],
) -> dict[str, Any]:
    ret_dict = {}
    for (_, sub_name), sub_dict in d.items():
        small_sub_name = sub_name.split(".")[-1]
        ret_dict.update({f"{small_sub_name}.{k}": v for k, v in sub_dict.items()})
    return ret_dict


if __name__ == "__main__":
    with gin.unlock_config():
        gin.parse_config_files_and_bindings(
            config_files=["atari.gin"],
            bindings=None,
        )
        EXPERIMENT.log_parameters(flatten_gin_config(gin.config._CONFIG))
        train()
