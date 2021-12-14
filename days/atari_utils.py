from collections import deque

import gym
import numpy as np
from gym import spaces
import cv2

# all of this is stolen from pfrl library

cv2.ocl.setUseOpenCL(False)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, channel_order="chw"):
        """Warp frames to 84x84 as done in the Nature paper and later work.

        To use this wrapper, OpenCV-Python is required.
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            "hwc": (self.height, self.width, 1),
            "chw": (1, self.height, self.width),
        }
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=shape[channel_order],
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame.reshape(self.observation_space.low.shape)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order="chw"):
        """Stack k last frames.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.stack_axis = {"hwc": 2, "chw": 0}[channel_order]
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=orig_obs_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=self.stack_axis)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Divide frame values by 255.0 and return them as np.float32.

    Especially, when the original env.observation_space is np.uint8,
    this wrapper converts frame values into [0.0, 1.0] of dtype np.float32.
    """
    def __init__(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        gym.ObservationWrapper.__init__(self, env)

        self.scale = 255.0

        orig_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=self.observation(orig_obs_space.low),
            high=self.observation(orig_obs_space.high),
            dtype=np.float32,
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / self.scale


def wrap_atari_env(env):
    return FrameStack(ScaledFloatFrame(WarpFrame(env)), 4)
