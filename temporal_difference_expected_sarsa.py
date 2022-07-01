
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


def expected_sarsa(env, iterations=10000):
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    policy = defaultdict(int)
    alpha = 0.5
    gamma = 0.95
    epsilon = 1
    decay = 1/iterations

    for iter in tqdm.tqdm(range(iterations)):
        old_state = env.reset()
        done = False

        while not done:
            prob_random_action = np.random.randn()
            if prob_random_action <= epsilon:
                action = env.action_space.sample()
            else:
                action = policy[old_state]

            state, reward, done, info = env.step(action)

            q_values = [Q[state][a] for a in range(n_actions)]
            max_q = np.max(q_values)
            count_max_q = q_values.count(max_q)
            if count_max_q > 1:
                best_actions = [a for a in range(n_actions) if q_values[a] == max_q]
                action_index = np.random.choice(best_actions)
            else:
                action_index = q_values.index(max_q)

            action_probs = [epsilon/n_actions for _ in range(n_actions)]
            best_action = action_index
            action_probs[best_action] += (1 - epsilon)

            expected_Q = 0
            for a in range(n_actions):
                expected_Q += Q[state][a]*action_probs[a]

            Q[old_state][action] += alpha * \
                (reward + gamma*expected_Q - Q[old_state][action])
            policy[old_state] = np.argmax(
                [Q[old_state][a] for a in range(n_actions)])
            old_state = state
        epsilon -= decay
        epsilon = max(0.1,epsilon)

    return Q, policy


def games_trial(env, policy, no_of_games=1000):
    count = 0
    for games in tqdm.tqdm(range(no_of_games)):
        done = False
        state = env.reset()
        while not done:
            action = policy[state]
            state, reward, done, info = env.step(action)

            if reward == 1:
                count += 1

    print(" No of games won:", count, "out of", no_of_games)


if __name__ == '__main__':
    Q, policy = expected_sarsa(env, iterations=100000)
    print(policy)

    V = np.array([np.max([Q[state][action] for action in range(n_actions)])
                  for state in range(n_states)])
    print(V.reshape(semi_states, semi_states))
    games_trial(env, policy, no_of_games=100)
    env.close()
