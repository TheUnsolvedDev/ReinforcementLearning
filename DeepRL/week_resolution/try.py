import gymnasium as gym
import numpy as np
import cv2

env = gym.make('PongNoFrameskip-v4')
obs = env.reset()
print(obs[0].shape)

from agents.params import *

env = make_env('PongNoFrameskip-v4')
obs = env.reset()
print(np.array(obs[0]).T.shape)

