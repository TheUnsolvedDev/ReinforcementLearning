
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gym
import sys
import tqdm

env = gym.make('FrozenLake-v1')
env = env.unwrapped

n_actions = env.action_space.n
n_states = env.observation_space.n
semi_states = int(n_states ** 0.5)


def temporal_difference_zero(env, iterations=10000):
    state_values = defaultdict(float)
    alpha = 0.01
    gamma = 0.95

    for iter in tqdm.tqdm(range(iterations)):
        old_state = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            state_values[old_state] += alpha * \
                (reward + gamma*state_values[state] - state_values[old_state])
            old_state = state

    return state_values


if __name__ == '__main__':
    state_values = temporal_difference_zero(env, iterations=5000000)
    state_values = np.array([state_values[i] for i in state_values.keys()]).reshape(
        semi_states, semi_states)
    print(state_values)
