from comet_ml import Experiment

import sys
from math import prod

import gin
import torch
import numpy as np
from torch import nn
from torch.optim import Adam, lr_scheduler
import gym
from tqdm import tqdm


def run_episode(env, net, eps, render=False):
    obs = env.reset()

    states = []
    rewards = []
    choices = []

    with torch.no_grad():
        while True:
            if render:
                env.render()
            states.append(obs)
            if torch.rand(()) < eps:
                choice = env.action_space.sample()
            else:
                expected_reward = net(torch.tensor(obs))
                choice = torch.argmax(expected_reward).numpy()
            choices.append(choice)
            obs, reward, done, _ = env.step(choice)
            rewards.append(reward)
            if done:
                break

    return states, rewards, choices


@gin.configurable
def train_dqn(experiment_name, env_id, gamma, episodes, start_eps, end_eps,
              hidden_size, start_lr, end_lr, train_frac_per_episode):
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="gDeuTHDCxQ6xdsXnvzkWsvDEb",
        project_name=f"train_dqn",
        workspace="rgreenblatt",
    )
    experiment.set_name(experiment_name)

    env = gym.make(env_id)
    net = nn.Sequential(
        nn.Linear(prod(env.observation_space.shape), hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, env.action_space.n),
    )
    loss_fn = nn.MSELoss(reduction='sum')

    optim = Adam(net.parameters(), lr=start_lr)
    scheduler = lr_scheduler.ExponentialLR(optim, gamma=(end_lr / start_lr)**(1 / episodes))

    losses = []
    durations = []

    for eps_idx in tqdm(range(episodes)):
        eps = (end_eps - start_eps) * eps_idx / episodes + start_eps
        states, rewards, choices = run_episode(env, net, eps)

        total_reward = sum(rewards)

        # maybe clean way to do this as a vectorized operation?
        for j in reversed(range(len(rewards) - 1)):
            rewards[j] += gamma * rewards[j + 1]

        episode_len = len(rewards)

        states = torch.tensor(np.array(states))
        rewards = torch.tensor(rewards)
        choices = torch.tensor(np.array(choices))

        subset = torch.randperm(episode_len)[:round(train_frac_per_episode *
                                                    episode_len)]

        states = states[subset]
        rewards = rewards[subset]
        choices = choices[subset]

        loss = loss_fn(rewards,
                       net(states)[torch.arange(states.size(0)), choices])

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        experiment.log_metric("episode len", episode_len)
        experiment.log_metric("total reward", total_reward)
        experiment.log_metric("eps", eps)
        experiment.log_metric("lr", scheduler.get_last_lr()[0])

        losses.append(loss.item())
        durations.append(episode_len)

    run_episode(env, net, eps=0, render=True)


def main():
    gin.parse_config_file(sys.argv[1])
    train_dqn()


if __name__ == "__main__":
    main()
