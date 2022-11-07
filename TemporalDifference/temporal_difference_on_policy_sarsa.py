
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


def sarsa(env, iterations=10000):
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    policy = defaultdict(int)
    alpha = 0.01
    gamma = 0.95
    epsilon = 1
    decay = 1/iterations

    for iter in tqdm.tqdm(range(iterations)):
        old_state = env.reset()[0]
        done = False
        if np.random.randn() <= epsilon:
            old_action = env.action_space.sample()
        else:
            old_action = policy[old_state]
        while not done:
            state, reward, done, info, _ = env.step(old_action)
            if np.random.randn() <= epsilon:
                action = env.action_space.sample()
            else:
                action = policy[old_state]

            Q[old_state][old_action] += alpha * \
                (reward + gamma*Q[state][action] - Q[old_state][old_action])
            policy[old_state] = np.argmax(
                [Q[old_state][a] for a in range(n_actions)])
            old_state = state
            old_action = action
        epsilon -= decay

    return Q, policy


def games_trial(env, policy, no_of_games=1000):
    count = 0
    for games in tqdm.tqdm(range(no_of_games)):
        done = False
        state = env.reset()[0]
        while not done:
            action = policy[state]
            state, reward, done, info, _ = env.step(action)

            if reward == 1:
                count += 1

    print(" No of games won:", count, "out of", no_of_games)


if __name__ == '__main__':
    Q, policy = sarsa(env, iterations=100000)
    print(policy)

    # V = np.array([np.max([Q[state][action] for action in range(n_actions)]) for state in range(n_states)])
    # print(V.reshape(semi_states,semi_states))
    games_trial(env, policy, no_of_games=100)
    env.close()
