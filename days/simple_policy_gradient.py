from comet_ml import Experiment

import sys
from math import prod

from torch.optim import Adam
import numpy as np
import gym
import torch
from torch import nn
import gin
from tqdm import tqdm


class CastToFloat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.float)


@gin.configurable
class MLP(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size, use_value_function=False):
        super().__init__()

        self._base = nn.Sequential(CastToFloat(),
                                   nn.Linear(obs_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU())
        self._use_value_function = use_value_function
        if use_value_function:
            self._to_value = nn.Linear(hidden_size, 1)
        self._to_policy = nn.Sequential(nn.Linear(hidden_size, action_size),
                                        nn.LogSoftmax(dim=-1))

    def forward(self, x):
        base = self._base(x)

        policy = torch.distributions.Categorical(logits=self._to_policy(base))

        if self._use_value_function:
            return self._to_value(base), policy


@gin.configurable
def train_simple_policy(experiment_name, env_id, steps, lr, batch_size,
                        rewards_to_go, advantage_estimation):
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="gDeuTHDCxQ6xdsXnvzkWsvDEb",
        project_name="train_simple_policy_gradient",
        workspace="rgreenblatt",
        disabled=False,
    )
    experiment.set_name(experiment_name)

    # torch.cuda.init()
    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    env = gym.make(env_id)

    net = MLP(prod(env.observation_space.shape),
              env.action_space.n, use_value_function=advantage_estimation).to(device=device)
    optim = Adam(net.parameters(), lr=lr)

    bar = tqdm(total=steps, disable=False)
    step = 0
    while step < steps:
        all_obs = []
        all_actions = []
        all_weights = []

        eps_rewards = []

        avg_reward = 0
        avg_eps_len = 0
        n_eps = 0

        obs = env.reset()
        while True:
            with torch.no_grad():
                all_obs.append(obs)
                output = net(torch.tensor(obs, device=device))
                if advantage_estimation:
                    policy = output[1]
                else:
                    policy = output
                action = policy.sample()
                all_actions.append(action)
                obs, reward, done, _ = env.step(action.cpu().item())
            eps_rewards.append(reward)
            step += 1
            bar.update(1)

            if done:
                obs = env.reset()
                total_reward = sum(eps_rewards)
                if rewards_to_go:
                    for i in reversed(range(len(eps_rewards))):
                        eps_rewards[i] = eps_rewards[i] + (
                            eps_rewards[i + 1] if i + 1 < len(eps_rewards) else 0)
                    all_weights += eps_rewards
                else:
                    total_reward = sum(eps_rewards)
                    all_weights += [total_reward] * len(eps_rewards)

                avg_reward += total_reward
                avg_eps_len += len(eps_rewards)
                n_eps += 1

                eps_rewards = []
                if len(all_obs) >= batch_size:
                    break

        avg_reward = avg_reward / n_eps
        avg_eps_len = avg_eps_len / n_eps

        weights = torch.tensor(all_weights, device=device)
        actions = torch.tensor(all_actions, device=device)
        dist = net(torch.tensor(np.array(all_obs), device=device))
        loss = -(weights * dist.log_prob(actions)).mean()

        experiment.log_metric("loss", loss.cpu().item(), step=step)
        experiment.log_metric("avg reward", total_reward, step=step)
        experiment.log_metric("avg episode length", avg_eps_len, step=step)

        optim.zero_grad()
        loss.backward()
        optim.step()

    obs = env.reset()
    while True:
        with torch.no_grad():
            action = net(torch.tensor(obs, device=device)).sample()
            obs, reward, done, _ = env.step(action.cpu().item())
            env.render()

        if done:
            break


def main():
    gin.parse_config_file(sys.argv[1])
    train_simple_policy()


if __name__ == "__main__":
    main()
