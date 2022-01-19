from __future__ import nested_scopes
from comet_ml import Experiment

import sys
from collections import deque
from math import prod
import itertools

import numpy as np
import gin
import torch
from torch import nn
from torch.optim import Adam
import gym
from tqdm import tqdm
from einops.layers.torch import Rearrange

from days.atari_wrappers import AtariWrapper


def make_choice(env, eps, net, obs, device):
    if torch.rand(()) < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            expected_reward = net(torch.tensor(obs, device=device).unsqueeze(0))
            return torch.argmax(expected_reward).cpu().numpy()


def run_eval_episode(count,
                     env,
                     net,
                     eps,
                     device,
                     save=False):
    assert count > 0
    for _ in range(count):
        obs = env.reset()
        video_recorder = None
        if save:
            video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env)
            print()
            print(f"recording to path: {video_recorder.path}")
        with torch.no_grad():
            while True:
                if video_recorder is not None:
                    video_recorder.capture_frame()
                obs, reward, done, _ = env.step(make_choice(env, eps, net, obs, device))
                if done:
                    break
        if video_recorder is not None:
            video_recorder.close()


class PixelByteToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.to(torch.float) / 255.0


    def forward(self, x):
        adv_v = self._advantage_side(x)
        adv_v = adv_v - adv_v.mean()

        value_v = self._value_side(x)

        return adv_v + value_v


def atari_model(obs_n_channels, action_size):
    return nn.Sequential(Rearrange("n h w c -> n c h w"), PixelByteToFloat(),
                         nn.Conv2d(obs_n_channels, 32, 8, stride=4), nn.ReLU(),
                         nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                         nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(3136, action_size))


class CastToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float)


@gin.configurable
def mlp_model(obs_size, action_size, hidden_size):
    return nn.Sequential(CastToFloat(), nn.Linear(obs_size, hidden_size),
                         nn.ReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.ReLU(), nn.Linear(hidden_size, action_size))


def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return:
    """
    def func(progress: float) -> float:
        if progress > end_fraction:
            return end
        else:
            return start + progress * (end - start) / end_fraction

    return func


@gin.configurable
def train_dqn(env_id,
              gamma,
              steps,
              start_eps,
              end_eps,
              exploration_frac,
              lr,
              train_freq,
              batch_size,
              buffer_size,
              multi_step_n,
              start_training_step=0):

    torch.cuda.init()
    DEVICE = torch.device('cuda:0')

    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = AtariWrapper(env)
        eval_env = AtariWrapper(eval_env, clip_reward=False)
        get_net = lambda: atari_model(env.observation_space.shape[-1], env.
                                      action_space.n)
    else:
        get_net = lambda: mlp_model(prod(env.observation_space.shape), env.
                                    action_space.n)

    net = get_net().to(DEVICE)
    get_params = lambda: net.parameters()

    optim = Adam(get_params(), lr=lr)
    loss_fn = nn.MSELoss()

    eps_sched = get_linear_fn(start_eps, end_eps, exploration_frac)

    obs = env.reset()

    buffer = deque([], maxlen=buffer_size)

    for step in tqdm(range(steps), disable=False):
        eps = eps_sched(step / steps)

        choice = make_choice(env, eps, net, obs, DEVICE)
        new_obs, reward, done, _ = env.step(choice)
        buffer.append((obs, choice, reward, done))
        obs = new_obs

        if done:
            obs = env.reset()

        if step >= start_training_step and (step + 1) % train_freq == 0:
            idxs = torch.randperm(len(buffer) - multi_step_n)[:batch_size] 
            # don't sample idxs that are too recent; can't update them yet

            obs_batch = [] 
            rewards = []
            choices = []
            dones = []
            next_obs = []

            # Probably this could be paralleled
            for idx in idxs:
                obs_b, choice_b, _, _ = buffer[idx]
                obs_batch.append(obs_b)
                choices.append(choice_b)
                total_reward_update = 0
                done_any = False
                for i in range(multi_step_n):
                    _, _, reward_b, done_b = buffer[idx + i]
                    total_reward_update += gamma**i * reward_b
                    if done_b:
                        done_any = True
                        break
                rewards.append(total_reward_update)
                dones.append(done_any)
                next_obs.append(buffer[idx + multi_step_n][0])

            obs_batch = torch.tensor(np.array(obs_batch), device=DEVICE)
            choices = torch.tensor(np.array(choices), device=DEVICE)
            rewards = torch.tensor(np.array(rewards),
                                   device=DEVICE,
                                   dtype=torch.float)
            dones = torch.tensor(np.array(dones), device=DEVICE)
            next_obs = torch.tensor(np.array(next_obs), device=DEVICE)

            with torch.no_grad():
                next_obs_actions = net(next_obs).argmax(dim=-1)
                targets = rewards + dones.logical_not() * (
                    gamma**multi_step_n * net(next_obs)[
                        torch.arange(idxs.size(0)), next_obs_actions])
                # Only include Q(S', A') if not within n of game completion

            actual = net(obs_batch)[torch.arange(idxs.size(0)),
                                            choices]
            loss = loss_fn(targets, actual)
            optim.zero_grad()
            loss.backward()

            optim.step()
    run_eval_episode(1, eval_env, net, eps=0, device=DEVICE, save=True)


def main():
    gin.parse_config_file(sys.argv[1])
    train_dqn()


if __name__ == "__main__":
    main()
