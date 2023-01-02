
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


def state_value_function(env, state, action, state_values, gamma):
    return np.sum([prob*(reward+gamma*state_values[next_state]) for
                   prob, next_state, reward, done in env.P[state][action]])


def value_iteration(env, iteration=100000, threshold=0.0001, gamma=0.99):
    state_values = np.zeros(n_states, dtype=np.float32)
    
    for iter in tqdm.tqdm(range(iteration)):
        delta = 0
        for state in range(n_states):
            old_value = state_values[state]
            state_values[state] = np.max([state_value_function(
                env, state, action, state_values, gamma) for action in range(n_actions)])
            delta = max(delta, np.abs(old_value - state_values[state]))
        if delta < threshold:
            break

    return state_values

def best_action(env,state,state_values,gamma):
    return np.argmax([state_value_function(env,state,action,state_values,gamma) for action in range(n_actions)])


def game_simulate(env,state_values, trials=10000):
    games_won = 0
    for trial in tqdm.tqdm(range(trials)):
        done = False
        state = env.reset()

        while not done:
            action = best_action(env,state,state_values,gamma = 0.99)
            state, reward, done, info = env.step(action)

            if reward == 1:
                games_won += 1
    print("Winning rate is", (games_won/trials)*100)


if __name__ == '__main__':
    state_values = value_iteration(env)
    print(state_values.reshape(semi_states, semi_states))
    game_simulate(env,state_values)
    env.close()
    
