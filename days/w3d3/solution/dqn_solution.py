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


def make_choice(env, eps, net, obs, DEVICE):
    if torch.rand(()) < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            expected_reward = net(
                torch.tensor(obs, device=DEVICE).unsqueeze(0))
            return torch.argmax(expected_reward).cpu().numpy()


def run_eval_episode(count,
                     env,
                     net,
                     eps,
                     DEVICE,
                     experiment=None,
                     step=None,
                     save=False):
    assert count > 0
    total_starting_q = 0
    total_reward = 0
    total_episode_len = 0
    for _ in range(count):
        obs = env.reset()

        video_recorder = None
        if save:
            video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(
                env)

            print()
            print(f"recording to path: {video_recorder.path}")
        with torch.no_grad():
            total_starting_q += net(
                torch.tensor(obs,
                             device=DEVICE).unsqueeze(0)).max().cpu().item()

            while True:
                if video_recorder is not None:
                    video_recorder.capture_frame()
                obs, reward, done, _ = env.step(
                    make_choice(env, eps, net, obs, DEVICE))
                total_reward += reward
                total_episode_len += 1
                if done:
                    break
        if video_recorder is not None:
            video_recorder.close()

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


class PixelByteToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.to(torch.float) / 255.0


class DuelingHead(nn.Module):
    def __init__(self, input_hidden_size, action_size, dueling_hidden_size):
        super().__init__()

        self._advantage_side = nn.Sequential(
            nn.Linear(input_hidden_size, dueling_hidden_size), nn.ReLU(),
            nn.Linear(dueling_hidden_size, action_size))
        self._value_side = nn.Sequential(
            nn.Linear(input_hidden_size, dueling_hidden_size), nn.ReLU(),
            nn.Linear(dueling_hidden_size, 1))

    def forward(self, x):
        adv_v = self._advantage_side(x)
        adv_v = adv_v - adv_v.mean()

        value_v = self._value_side(x)

        return adv_v + value_v

@gin.configurable
def mlp_model(obs_size, action_size, hidden_size):
    return nn.Sequential(CastToFloat(), nn.Linear(obs_size, hidden_size),
                         nn.ReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.ReLU(), dqn_head(hidden_size, action_size))

@gin.configurable
def dqn_head(hidden_size, action_size, use_dueling, dueling_hidden_size=256):
    if use_dueling:
        return DuelingHead(hidden_size, action_size, dueling_hidden_size)
    else:
        return nn.Linear(hidden_size, action_size)


def atari_model(obs_n_channels, action_size):
    return nn.Sequential(Rearrange("n h w c -> n c h w"), PixelByteToFloat(),
                         nn.Conv2d(obs_n_channels, 32, 8, stride=4), nn.ReLU(),
                         nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                         nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                         nn.Flatten(), dqn_head(3136, action_size))


class CastToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float)





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
def train_dqn(experiment_name,
              env_id,
              gamma,
              steps,
              start_eps,
              end_eps,
              exploration_frac,
              lr,
              train_freq,
              batch_size,
              buffer_size,
              eval_freq,
              use_double_dqn,
              multi_step_n,
              eval_count=1,
              start_training_step=0):
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="gDeuTHDCxQ6xdsXnvzkWsvDEb",
        project_name="train_dqn",
        workspace="rgreenblatt",
        disabled=False,
    )
    experiment.set_name(experiment_name)

    torch.cuda.init()
    DEVICE = torch.device('cuda:0')
    # DEVICE = torch.device('cpu')

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

    if use_double_dqn:
        nets = [get_net().to(DEVICE) for _ in range(2)]
        get_params = lambda: itertools.chain(
            *[net.parameters() for net in nets])
    else:
        net = get_net().to(DEVICE)
        get_params = lambda: net.parameters()

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

    optim = Adam(get_params(), lr=lr)
    loss_fn = nn.MSELoss()

    eps_sched = get_linear_fn(start_eps, end_eps, exploration_frac)

    obs = env.reset()

    with torch.no_grad():
        starting_q = both_nets(torch.tensor(obs,
                                            device=DEVICE).unsqueeze(0)).max()

    buffer = deque([], maxlen=buffer_size)
    episode_len = 0
    total_reward = 0

    for step in tqdm(range(steps), disable=False):
        eps = eps_sched(step / steps)

        choice = make_choice(env, eps, both_nets, obs, DEVICE)
        new_obs, reward, done, _ = env.step(choice)
        buffer.append((obs, choice, reward, done))
        obs = new_obs

        episode_len += 1
        total_reward += reward

        if done:
            obs = env.reset()
            with torch.no_grad():
                starting_q = both_nets(
                    torch.tensor(obs, device=DEVICE).unsqueeze(0)).max()

            experiment.log_metric("episode len", episode_len, step=step)
            experiment.log_metric("total reward", total_reward, step=step)
            experiment.log_metric("eps", eps, step=step)
            experiment.log_metric("starting Q", starting_q, step=step)

            episode_len = 0
            total_reward = 0

        if step >= start_training_step and (step + 1) % train_freq == 0:
            idxs = torch.randperm(len(buffer) - multi_step_n)[:batch_size]

            obs_batch = []
            rewards = []
            choices = []
            dones = []
            next_obs = []

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
                next_obs_actions = split_nets(next_obs,
                                              flip=False).argmax(dim=-1)
                targets = rewards + dones.logical_not() * (
                    gamma**multi_step_n * split_nets(next_obs, flip=True)[
                        torch.arange(idxs.size(0)), next_obs_actions])

            actual = split_nets(obs_batch,
                                flip=False)[torch.arange(idxs.size(0)),
                                            choices]
            loss = loss_fn(targets, actual)
            experiment.log_metric("loss", loss.cpu().item(), step=step)
            optim.zero_grad()
            loss.backward()

            optim.step()

        if (step + 1) % eval_freq == 0:
            run_eval_episode(eval_count,
                             eval_env,
                             both_nets,
                             eps=0.05,
                             device=DEVICE,
                             experiment=experiment,
                             step=step,
                             save=False)

    run_eval_episode(1, eval_env, both_nets, eps=0, device=DEVICE, save=True)


def main():
    gin.parse_config_file(sys.argv[1])
    train_dqn()


if __name__ == "__main__":
    main()
