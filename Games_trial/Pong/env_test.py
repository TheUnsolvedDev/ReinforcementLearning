import numpy as np
import gymnasium as gym
import tensorflow as tf
import matplotlib.pyplot as plt

from params import *

env = gym.make('ALE/Pong-v5')
# test_env = gym.make('PongNoFrameskip-v4', render_mode='human')

# Get the action space and state space dimensions
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

if __name__ == '__main__':
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info, truncated = env.step(action)
        print(state[34:193,:,1].shape)
        plt.imshow(state[34:193,:,1])
        plt.show()
        print(reward, action)
        # env.render()
