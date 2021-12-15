from comet_ml import Experiment

import sys
from math import prod
import itertools

import numpy as np
import gin
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
import gym
from tqdm import tqdm

from days.atari_utils import wrap_atari_env


def make_choice(env, eps, net, obs, device):
    if torch.rand(()) < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            expected_reward = net(
                torch.tensor(obs, device=device).unsqueeze(0))
            return torch.argmax(expected_reward).cpu().numpy()


def run_eval_episode(count,
                     env,
                     net,
                     eps,
                     device,
                     experiment=None,
                     step=None,
                     render=False):
    assert count > 0
    total_starting_q = 0
    total_reward = 0
    total_episode_len = 0
    for _ in range(count):
        obs = env.reset()

        total_starting_q += net(torch.tensor(
            obs, device=device).unsqueeze(0)).max().cpu().item()

        with torch.no_grad():
            while True:
                if render:
                    env.render()
                obs, reward, done, _ = env.step(
                    make_choice(env, eps, net, obs, device))
                total_reward += reward
                total_episode_len += 1
                if done:
                    break

    prefix = "" if count == 1 else "avg "
    if experiment is not None:
        experiment.log_metric(f"eval {prefix}episode len",
                              total_episode_len / count,
                              step=step)
        experiment.log_metric(f"eval {prefix}total reward",
                              total_reward / count,
                              step=step)
        experiment.log_metric(f"eval {prefix}starting q",
                              total_starting_q / count,
                              step=step)


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
def train_dqn(experiment_name,
              env_id,
              gamma,
              steps,
              start_eps,
              end_eps,
              start_lr,
              end_lr,
              train_freq,
              batch_size,
              buffer_size,
              eval_freq,
              use_double_dqn,
              eval_count=1):
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
    # device = torch.device('cpu')

    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = wrap_atari_env(env)
        eval_env = wrap_atari_env(eval_env)
        get_net = lambda: atari_model(env.observation_space.shape[0], env.
                                      action_space.n)
    else:
        get_net = lambda: mlp_model(prod(env.observation_space.shape), env.
                                    action_space.n)

    if use_double_dqn:
        nets = [get_net().to(device) for _ in range(2)]
        params = itertools.chain(*[net.parameters() for net in nets])
    else:
        net = get_net().to(device)
        params = net.parameters()

    def both_nets(inp):
        if use_double_dqn:
            return (nets[0](inp) + nets[1](inp)) / 2
        else:
            return net(inp)

    def split_nets(inp, flip):
        if use_double_dqn:
            half_s = inp.size(0) // 2
            l, r = 0, 1
            if flip:
                r, l = l, r

            return torch.concat((nets[l](inp[:half_s]), nets[r](inp[half_s:])))
        else:
            return net(inp)

    optim = Adam(params, lr=start_lr)
    loss_fn = nn.MSELoss(reduction='mean')

    scheduler = lr_scheduler.ExponentialLR(optim,
                                           gamma=(end_lr / start_lr)**(1 /
                                                                       steps))

    obs = env.reset()

    with torch.no_grad():
        starting_q = both_nets(torch.tensor(obs,
                                            device=device).unsqueeze(0)).max()

    buffer = []
    episode_len = 0
    total_reward = 0

    for step in tqdm(range(steps), disable=False):
        eps = (end_eps - start_eps) * step / steps + start_eps

        choice = make_choice(env, eps, both_nets, obs, device)
        new_obs, reward, done, _ = env.step(choice)
        buffer.append((obs, choice, reward, done))
        obs = new_obs

        episode_len += 1
        total_reward += reward

        if done:
            obs = env.reset()
            with torch.no_grad():
                starting_q = both_nets(
                    torch.tensor(obs, device=device).unsqueeze(0)).max()

            experiment.log_metric("episode len", episode_len, step=step)
            experiment.log_metric("total reward", total_reward, step=step)
            experiment.log_metric("eps", eps, step=step)
            experiment.log_metric("lr", scheduler.get_last_lr()[0], step=step)
            experiment.log_metric("starting Q", starting_q, step=step)

            episode_len = 0
            total_reward = 0

        if (step + 1) % train_freq == 0:
            idxs = torch.randperm(len(buffer) - 1)[:batch_size]

            obs_batch = []
            rewards = []
            choices = []
            dones = []
            next_obs = []

            for idx in idxs:
                obs_b, choice_b, reward_b, done_b = buffer[idx]
                obs_batch.append(obs_b)
                choices.append(choice_b)
                rewards.append(reward_b)
                dones.append(done_b)
                next_obs.append(buffer[idx + 1][0])
                next_obs.append(buffer[idx + multi_step_n][0])

            obs_batch = torch.tensor(np.array(obs_batch), device=device)
            choices = torch.tensor(np.array(choices), device=device)
            rewards = torch.tensor(np.array(rewards),
                                   device=device,
                                   dtype=torch.float)
            dones = torch.tensor(np.array(dones), device=device)
            next_obs = torch.tensor(np.array(next_obs), device=device)

            targets = rewards + dones.logical_not() * (
                gamma * split_nets(next_obs, flip=True).max(dim=-1).values)

            loss = loss_fn(
                targets,
                split_nets(obs_batch, flip=False)[torch.arange(idxs.size(0)),
                                                  choices])
            experiment.log_metric("loss", loss.cpu().item(), step=step)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            target_size = buffer_size - train_freq
            to_cut = len(buffer) - target_size
            del buffer[:to_cut]

        if (step + 1) % eval_freq == 0:
            run_eval_episode(eval_count,
                             eval_env,
                             both_nets,
                             eps=0.05,
                             device=device,
                             experiment=experiment,
                             step=step,
                             render=True)

    run_eval_episode(eval_count,
                     eval_env,
                     both_nets,
                     eps=0,
                     device=device,
                     render=True)


def main():
    gin.parse_config_file(sys.argv[1])
    train_dqn()


if __name__ == "__main__":
    main()
