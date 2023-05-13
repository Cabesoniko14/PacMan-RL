import gym#initial environment
from gym.spaces import Box
from gym import spaces
import numpy as np
import os
from collections import deque
import cv2

env = gym.make("ALE/MsPacman-v5", render_mode='human')

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        '''
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        '''
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs
    
env = WarpFrame(env)
print('Hola')
env.action_space.seed(42) # Seed aleatoria

observacion, info = env.reset(seed=42) # Resetear el environment a uno aleatorio, pero siempre el mismo

env.reset()
for _ in range(100):  # Por cada paso, que son 100, ejecuta lo siguiente
    observacion, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    print(reward)

    if terminated or truncated:
        observacion, info = env.reset()

env.close()

print(observacion.shape)
