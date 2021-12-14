from comet_ml import Experiment

import time
import sys
from math import prod
from multiprocessing import Pool
from functools import partial

import gin
import torch
import numpy as np
from torch import nn
from torch.optim import Adam, lr_scheduler
import gym
from tqdm import tqdm

from days.atari_utils import wrap_atari_env


def run_episode(env, net, eps, device, render=False):
    obs = env.reset()

    states = []
    rewards = []
    choices = []

    def call(obs):
        return net(torch.tensor(obs).unsqueeze(0).to(device))

    starting_q = call(obs).max()

    with torch.no_grad():
        while True:
            if render:
                env.render()
            states.append(obs)
            if torch.rand(()) < eps:
                choice = env.action_space.sample()
            else:
                expected_reward = call(obs)
                choice = torch.argmax(expected_reward).cpu().numpy()
            choices.append(choice)
            obs, reward, done, _ = env.step(choice)
            rewards.append(reward)
            if done:
                break

    return starting_q, states, rewards, choices


def atari_model(obs_n_channels, action_size):
    return nn.Sequential(
        nn.Conv2d(obs_n_channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, action_size),
    )

class CastToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float)

@gin.configurable
def mlp_model(obs_size, action_size, hidden_size):
    return nn.Sequential(
        CastToFloat(),
        nn.Linear(obs_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size),
    )


@gin.configurable
def train_dqn(experiment_name, env_id, gamma, episodes, start_eps, end_eps,
              start_lr, end_lr, train_frac_per_episode):
    # # Create an experiment with your api key
    experiment = Experiment(
        api_key="gDeuTHDCxQ6xdsXnvzkWsvDEb",
        project_name=f"train_dqn",
        workspace="rgreenblatt",
        disabled=False,
    )
    experiment.set_name(experiment_name)

    torch.cuda.init()

    device = torch.device('cuda:0')

    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = wrap_atari_env(env)
        net = atari_model(env.observation_space.shape[0], env.action_space.n)
    else:
        net = mlp_model(prod(env.observation_space.shape), env.action_space.n)
    net = net.to(device)
    loss_fn = nn.MSELoss(reduction='none')

    optim = Adam(net.parameters(), lr=start_lr)
    scheduler = lr_scheduler.ExponentialLR(optim,
                                           gamma=(end_lr /
                                                  start_lr)**(1 / episodes))

    for eps_idx in tqdm(range(episodes), disable=False):
        eps = (end_eps - start_eps) * eps_idx / episodes + start_eps
        starting_q, states, rewards, choices = run_episode(
            env, net, eps, device)

        total_reward = sum(rewards)

        # maybe clean way to do this as a vectorized operation?
        for j in reversed(range(len(rewards) - 1)):
            rewards[j] += gamma * rewards[j + 1]

        discounted_start_reward = rewards[0]

        episode_len = len(rewards)

        states = torch.tensor(np.array(states), device=device)
        rewards = torch.tensor(rewards, device=device)
        choices = torch.tensor(np.array(choices), device=device)

        subset = torch.randperm(episode_len,
                                device=device)[:round(train_frac_per_episode *
                                                      episode_len)]

        states = states[subset]
        rewards = rewards[subset]
        choices = choices[subset]

        unreduced_loss = loss_fn(
            rewards,
            net(states)[torch.arange(states.size(0)), choices])

        optim.zero_grad()
        unreduced_loss.sum().backward()
        optim.step()
        scheduler.step()

        experiment.log_metric("episode len", episode_len)
        experiment.log_metric("total reward", total_reward)
        experiment.log_metric("eps", eps)
        experiment.log_metric("lr", scheduler.get_last_lr()[0])
        experiment.log_metric("mean loss", unreduced_loss.mean())
        experiment.log_metric("starting Q", starting_q)
        experiment.log_metric("discounted start reward", discounted_start_reward)

    run_episode(env, net, eps=0, device=device, render=True)


def main():
    gin.parse_config_file(sys.argv[1])
    train_dqn()


if __name__ == "__main__":
    main()
