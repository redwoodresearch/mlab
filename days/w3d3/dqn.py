import collections
import dataclasses
import random
from typing import Optional

import gym
import numpy as np
import torch as t
from comet_ml import Experiment
from torch import nn, optim
from tqdm import tqdm

from video_recorder import VideoRecorder

from copy import deepcopy


def get_eps_greedy_action(
    env: gym.Env,
    eps: float,
    net: nn.Module,
    obs: np.ndarray,
) -> int:
    """Returns one epsilon-greedy action."""

    # With probability eps return uniformly random action.
    if t.rand(1) < eps:
        return env.action_space.sample()

    q_vals = net(t.tensor(obs, dtype=net.dtype, device=net.device))
    assert len(q_vals.shape) == 1

    return t.argmax(q_vals).item()


def evaluate(
    model: nn.Module,
    env: gym.Env,
    eps: float = 0,
    video_path: Optional[str] = None,
) -> float:
    """
    Runs model for one episode.
    Returns reward.
    """
    if video_path is not None:
        recorder = VideoRecorder(env, video_path)

    with t.no_grad():
        steps = 0
        obs = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            if video_path is not None:
                recorder.capture_frame()

            obs, reward, done, _ = env.step(
                get_eps_greedy_action(env=env, eps=eps, net=model, obs=obs)
            )
            total_reward += reward
            steps += 1

    if video_path is not None:
        recorder.close()
    return total_reward, steps


@dataclasses.dataclass
class BufferEntry:
    obs: np.ndarray
    obs_next: np.ndarray
    action: int
    reward: float
    done: bool


def dqn_train(
    model: nn.Module,
    env: gym.Env,
    n_steps: int,
    batch_size: int,
    max_buffer_size: int = 10_000,
    train_freq: int = 10,
    eval_freq: int = 100,
    video_freq: int = 1000,
    eps_start: float = 0.5,
    eps_end: float = 0.05,
    gamma: float = 1.0,
    lr: float = 1e-3,
    seed: int = 42,
    experiment: Optional[Experiment] = None,
):
    opt = optim.Adam(model.parameters(), lr=lr)
    buffer = collections.deque(maxlen=max_buffer_size)

    perf_history: list[float] = []
    loss_history: list[float] = []
    obs: np.ndarray = env.reset()
    for i in tqdm(range(n_steps)):
        eps = eps_start * (1 - i / n_steps) + eps_end * (i / n_steps)
        action = get_eps_greedy_action(env=env, eps=eps, net=model, obs=obs)
        obs_next, reward, done, _ = env.step(action)
        if done:
            obs_next = env.reset()

        buffer.append(
            BufferEntry(
                obs=obs,
                obs_next=obs_next,
                action=action,
                reward=reward,
                done=done,
            )
        )
        obs = obs_next

        if i % train_freq == 0 and i >= batch_size:

            batch_entries: list[BufferEntry] = random.sample(buffer, batch_size)

            batch_obs = t.tensor(
                np.array([e.obs for e in batch_entries]),
                dtype=model.dtype,
                device=model.device,
            )
            batch_obs_next = t.tensor(
                np.array([e.obs_next for e in batch_entries]),
                dtype=model.dtype,
                device=model.device,
            )

            batch_terminal = t.tensor(
                [e.done for e in batch_entries], device=model.device, dtype=t.bool
            )
            batch_reward = t.tensor(
                [e.reward for e in batch_entries], device=model.device
            )

            with t.no_grad():
                q_next_best = t.amax(model(batch_obs_next), dim=-1)
                y = batch_reward + gamma * (~batch_terminal) * q_next_best

            action_idxs = t.tensor(
                [e.action for e in batch_entries], dtype=t.long, device=model.device
            )
            q = t.gather(model(batch_obs), 1, action_idxs.unsqueeze(dim=1)).flatten()
            loss = ((q - y) ** 2).mean()
            loss_history.append(loss.item())

            if experiment is not None:
                experiment.log_metric(name="loss", value=loss.item(), step=i)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if i % eval_freq == 0:
            fresh_env = deepcopy(env)
            fresh_env.seed(seed)
            fresh_env.action_space.seed(seed)

            video_path = "videos/dqn_atari_comet.mp4" if i % video_freq == 0 else None
            reward, rollout_len = evaluate(
                model,
                fresh_env,
                eps=0,
                video_path=video_path,
            )
            perf_history.append(reward)

            if experiment is not None:
                experiment.log_metric(name="reward", value=reward, step=i)
                experiment.log_metric(name="rollout_len", value=rollout_len, step=i)

                if video_path is not None:
                    experiment.log_asset(video_path)

    return loss_history, perf_history
