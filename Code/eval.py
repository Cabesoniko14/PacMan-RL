# Probar las thetas

# Import de bibliotecas

import gym
# %matplotlib inline --> para ipynb
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import cv2


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ----------------


class PacmanEnv(gym.Env):
    def __init__(self, render_mode="human", seed=None):
        if render_mode == "human":
            self.env = gym.make("ALE/MsPacman-v5", render_mode="human")
        else:
            self.env = gym.make("ALE/MsPacman-v5")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.seed = seed
        self.lives = self.env.ale.lives()
        
        # Dejamos lo demás igual, nos ayuda mucho que tengamos las posibles acciones
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # custom reward
        if reward > 0: # recibió algún tipo de reward
            
            if info['lives'] > 0:  # Si el número de vidas de pacman es positivo
                if reward == 10:  # En el caso por default, 10 de reward es por comer un puntito
                    reward = 25 # Si come un puntito, que el reward sea __
                elif reward == 50: # Si come un power pellet que activa el poder comer fantasmas
                    reward = 60
                elif reward == 200:  # En el caso por default, hay reward de 200 por comer un fantasma
                    reward = 100 # Si come un fantasma, cuánto queremos de reward
            else: # Si a pacman se le acabaron las vidas
                done = True  # Pone en true la bandera de que acabó el juego
                reward = -500  # Si perdemos, cuánto queremos que pierda de reward. Que le duela al pacman
                
        else:
            reward = -1  # Cuando camine pero no consiga nada, que busque la ruta óptima. Importante !
            
            # Castigo por perder vidas
            lives = info['lives']
            if self.env.ale.lives() < lives:
                # Agent lost a life:
                reward -= 1000

        return observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()

    def render(self, mode=None, render=True):
        if self.render_mode == 'human' and render:
            self.env.render(mode=mode)

    def close(self):
        self.env.close()



# ----------



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
    
    
# ----------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------

# DQN

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = x.float() / 255
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x


# ----------------------------------------------------------------------------------

# Load the saved policy network weights on CPU
policy_net = torch.load('/Users/javi/Desktop/PacMan-RL/Code/policy_net.pt', map_location=torch.device('cpu'))
policy_net.eval()


# Set up the environment for testing
test_env = PacmanEnv(render_mode = 'human')
test_env = WarpFrame(test_env)
test_env = gym.wrappers.FrameStack(test_env, 4)

# Initialize the state of the environment
state, info = test_env.reset()
state = torch.tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
state = state.squeeze(-1)

while True:
        
        action = policy_net(state).max(1)[1].view(1, 1)
        observation, reward, terminated, truncated, _ = test_env.step(action.item())
        reward = torch.tensor([reward], device='cpu')
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device='cpu').unsqueeze(0)
            next_state = next_state.squeeze(-1)


        # Move to the next state
        state = next_state

        if done:
            break

test_env.close()
   
