import random
import gym
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import Video
import video_recorder
import torch.nn.functional as F
import torch.nn as nn
import torch as t
import collections
import gin
import numpy as np
from einops import rearrange


gin.external_configurable(t.optim.Adam)

class MLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = self.fc2(x)
        x = self.fc3(F.relu(x))
        return x


class Atari(nn.Module):
    def __init__(self, n_action_space: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, n_action_space),
        )
    
    def forward(self, x):
        x = rearrange(x, 'h w c -> 1 c h w')
        return self.net(x).squeeze(0)


@gin.configurable
def make_choice(env, eps, model, obs, device):
    sample = t.rand(1)
    if sample < eps:
        return env.action_space.sample()
    else:
        q_pred = model(t.tensor(obs, dtype=t.float32).to(device))
        index = q_pred.argmax(dim=-1)
        return index.item()


@gin.configurable
def evaluate(model, env, step: int, n_episodes: int, experiment, record=False):
    video_name = f"{env.unwrapped.spec.id}_{step}"

    episodes_rewards = []
    for _ in range(n_episodes):
        if record:
            recorder = video_recorder.VideoRecorder(env, f"videos/{video_name}.mp4")

        obs = env.reset()
        done = False
        total_reward = 0
        with t.no_grad():
            while not done:
                env.render()
                if record:
                    recorder.capture_frame()
                action = make_choice(env, eps=0, model=model, obs=obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
        if record:
            recorder.close()
        env.reset()
        episodes_rewards.append(total_reward)
        # experiment.log_metric("eval_reward", total_reward, step=step)
    return np.mean(np.array(episodes_rewards))


@gin.configurable
def gradient_step(
    model, buffer, gamma: float, batch_size: int, device: str, optim: t.optim.Optimizer,
):
    # sample minibatch from buffer
    optim.zero_grad()
    batch = random.sample(buffer, k=batch_size)
    loss = 0
    for transition in batch:
        obs, action, reward, done, new_obs = transition
        obs = t.tensor(obs, dtype=t.float32).to(device)
        action = t.tensor(action, dtype=t.int32).to(device)
        reward = t.tensor(reward, dtype=t.float32).to(device)
        new_obs = t.tensor(new_obs, dtype=t.float32).to(device)
        q_pred = model(obs)
        y = reward
        if not done:
            with t.no_grad():
                max_q = t.max(model(new_obs), dim=-1).values
                y += gamma * max_q
        loss += t.square(y - q_pred[action])
    loss.backward()
    optim.step()


@gin.configurable
def train(
    n_steps,
    train_freq,
    model,
    env,
    eval_freq: int,
    batch_size: int,
    maxlen: int,
    optimizer: t.optim.Optimizer,
    epsilon_start: float,
    epsilon_end: float,
    experiment
):
    optim = optimizer(params=model.parameters())

    delta = (epsilon_start - epsilon_end) / n_steps
    eps = epsilon_start

    done = False
    buffer = collections.deque(maxlen=maxlen)
    obs = env.reset()
    print("Starting training")
    for step in range(n_steps):
        total_reward = 0
        action = make_choice(env=env, eps=eps, model=model, obs=obs)
        new_obs, reward, done, _ = env.step(action)
        buffer.append((obs, action, reward, done, new_obs))
        obs = new_obs
        if done:
            obs = env.reset()
            done = False
        total_reward += reward
        eps -= delta

        if step > batch_size and step % train_freq == 0:
            gradient_step(model, buffer, batch_size=batch_size, optim=optim)

        if step > 0 and step % eval_freq == 0:
            eval_out = evaluate(model, env, step, experiment=experiment)
            print(f"{step=}, reward: {eval_out}")
