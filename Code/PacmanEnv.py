import numpy as np
import gym
from gym import spaces

class PacmanEnv(gym.Env):
    def __init__(self, render_mode="human", seed=None):
        if render_mode == "human":
            self.env = gym.make("ALE/MsPacman-v5", render_mode="human")
        else:
            self.env = gym.make("ALE/MsPacman-v5")
        self.action_space = self.env.action_space
        self.seed = seed
        self.lives = self.env.ale.lives()
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def preprocess(self, observation):
        img = observation[1:168:2, : : 2, :] # crop and downsize
        img = img - 128 # normalize between -128 and 127
        img = img.astype('int8') # saves memory
        img = observation
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.preprocess(observation)
        # ... 
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = self.env.reset()
        observation = self.preprocess(observation)
        return observation