

import re
import tqdm
import numpy as np
import gym
import time
import sys

env = gym.make('FrozenLake-v1')
# env = gym.make('FrozenLake8x8-v1')
env = env.unwrapped

env.reset()
n_actions = 4
n_states = 16
# n_states = 64


def random_gameplay(trials=5):
    for plays in range(trials):
        done = False
        env.reset()

        while not done:
            env.render()
            actions = np.random.randint(0, 4)
            obs, reward, done, info = env.step(actions)
            time.sleep(0.3)
    env.close()


def value_iteration(env, iterations=50000, threshold=0.00001, gamma=0.95):
    state_values = np.zeros(n_states)
    new_state_values = state_values.copy()
    delta = 0
    for iter in tqdm.tqdm(range(iterations)):
        for state in range(n_states):
            action_values = []
            for action in range(n_actions):
                state_value = 0
                for index in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][index]
                    state_action_value = prob * \
                        (reward + gamma*state_values[next_state])
                    state_value += state_action_value
                action_values.append(state_value)
            best_action = np.argmax(np.array(action_values))
            new_state_values[state] = action_values[best_action]
            delta = max(delta, np.abs(
                state_value - new_state_values[best_action]))
        if delta < threshold and iter > 1000:
            break
        else:
            state_values = new_state_values.copy()
    return state_values


def optimal_policy(env, state_values, gamma=0.9):
    actions = np.zeros(n_states,dtype = np.int8)
    for state in range(n_states):
        action_values = []
        for action in range(n_actions):
            action_value = 0
            for index in range(len(env.P[state][action])):
                prob, next_state, reward, done = env.P[state][action][index]
                action_value += prob * \
                    (reward + gamma*state_values[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.array(action_values))
        actions[state] = best_action
    return actions


def play(env, policy, iterations=10):
    count = 0
    for iter in range(iterations):
        obs = env.reset()
        while True:
            env.render()
            action = policy[obs]
            obs, reward, done, _ = env.step(action)
            if done and reward == 1:
                count += 1
                print("Yay! you won!")
                break
            elif done and reward == 0:
                print("You! lost!")
                break
    env.close()

    print("You won!", count, "out of", iterations, "games.")


if __name__ == '__main__':
    # random_gameplay(6)
    state_values = value_iteration(env)
    print(state_values.reshape((4, 4)))
    policy = optimal_policy(env, state_values)
    print(policy.reshape((4, 4)))
    play(env, policy, iterations=10)

"""
4x4 gameplay

State Values
[[0.18047158 0.15475672 0.15347714 0.13254844]
 [0.20896709 0.         0.17643079 0.        ]
 [0.27045741 0.37465152 0.40367272 0.        ]
 [0.         0.50897995 0.72367364 0.        ]]
 
Action Values
[[0. 3. 0. 3.]
 [0. 0. 0. 0.]
 [3. 1. 0. 0.]
 [0. 2. 1. 0.]]
"""
