import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

hidden_size = 16

net = nn.Sequential(
    nn.Linear(4, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, 2),
)
loss_fn = nn.MSELoss(reduction='mean')
optim = Adam(net.parameters(), lr=1e-2)

discount = 0.95
episodes = 10000
eps = 0.05

obs_frac = 0.2

losses = []
durations = []

for i in range(episodes):
    obs = env.reset()

    states = []
    rewards = []
    choices = []
    was_eps = []

    with torch.no_grad():
        while True:
            states.append(obs)
            if torch.rand(()) < eps:
                was_eps.append(True)
                choice = env.action_space.sample()
            else:
                was_eps.append(False)
                expected_reward = net(torch.tensor(obs))
                choice = torch.argmax(expected_reward).numpy()
            choices.append(choice)
            obs, reward, done, _ = env.step(choice)
            rewards.append(reward)
            if done:
                break

    for j in reversed(range(len(rewards) - 1)):
        rewards[j] += discount * rewards[j + 1]

    episode_len = len(rewards)

    states = torch.tensor(np.array(states))
    rewards = torch.tensor(rewards)
    choices = torch.tensor(np.array(choices))

    if i == 9999:
        print(rewards)
        print(choices)
        print(net(states))
        print(net(states)[torch.arange(states.size(0)), choices])
        print(was_eps)

    subset = torch.randint(episode_len, (round(obs_frac * episode_len),))

    states = states[subset]
    rewards = rewards[subset]
    choices = choices[subset]

    loss = loss_fn(rewards, net(states)[torch.arange(states.size(0)), choices])

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 1000 == 0:
        print(episode_len)


    losses.append(loss.item())
    durations.append(episode_len)

plt.plot(losses)
plt.show()
plt.plot(durations)
plt.show()
