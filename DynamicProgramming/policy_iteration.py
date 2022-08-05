
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

def state_value_function(env, state, action, state_values, gamma=0.999):
    return np.sum([prob*(reward+gamma*state_values[next_state]) for
                   prob, next_state, reward, done in env.P[state][action]])


def policy_evaluation(env, state_values, policy, iterations=50000, threshold=0.00001, gamma=0.999):
    for iter in tqdm.tqdm(range(iterations)):
        delta = 0
        for state in range(n_states):
            old_value = state_values[state]
            state_value = 0
            action = policy[state]
            state_values[state] = state_value_function(
                env, state, action, state_values, gamma)
            delta = max(delta, np.abs(old_value - state_values[state]))

        if delta < threshold:
            break
    return state_values, policy, delta


def policy_improvement(env, state_values, policy, gamma):
    for state in range(n_states):
        policy[state] = np.argmax([state_value_function(env,
                                                        state, action, state_values, gamma) for action in range(n_actions)])
    return state_values, policy


def policy_iteration(env, gamma=0.999):
    state_values = np.zeros(n_states)
    policy = np.zeros(n_states)
    delta = np.inf

    converged = False
    while not converged:
        old_states = state_values.copy()
        old_policies = policy.copy()

        print('Converging Policy Evaluation The Delta status is:', delta)
        state_values, policy, delta = policy_evaluation(
            env, state_values, policy, gamma=gamma)

        print('Optimizing the policies')
        state_values, policy = policy_improvement(
            env, state_values, policy, gamma=gamma)

        all_similar = np.prod(old_policies == policy)
        if all_similar:
            converged = True

    return state_values, policy

def game_simulate(env,policy, trials=10000):
    games_won = 0
    for trial in tqdm.tqdm(range(trials)):
        done = False
        state = env.reset()

        while not done:
            action = policy[state]
            state, reward, done, info = env.step(action)

            if reward == 1:
                games_won += 1
    print("Winning rate is", (games_won/trials)*100)


if __name__ == '__main__':
    state_values,policy = policy_iteration(env)
    print(state_values.reshape(semi_states, semi_states))
    game_simulate(env,policy)
    env.close()
    