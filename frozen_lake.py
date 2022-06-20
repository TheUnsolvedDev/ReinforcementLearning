
import time
import tqdm
import numpy as np
import gym
import sys

# env = gym.make('FrozenLake-v1')
env = gym.make('FrozenLake8x8-v1')
env = env.unwrapped

env.reset()
n_actions = 4
# n_states = 16
n_states = 64
semi_states = int(n_states ** 0.5)


def random_gameplay(trials=5):
    for plays in range(trials):
        done = False
        env.reset()

        while not done:
            env.render()
            actions = np.random.randint(0, 4)
            obs, reward, done, info = env.step(actions)
    env.close()


def state_value_function(env, state, action, state_values, gamma=0.999):
    return np.sum([prob*(reward+gamma*state_values[next_state]) for
                   prob, next_state, reward, done in env.P[state][action]])


def value_iteration(env, iterations=50000, threshold=0.00001, gamma=0.99):
    state_values = np.zeros(n_states)
    delta = 0
    for iter in tqdm.tqdm(range(iterations)):
        for state in range(n_states):
            old_value = state_values[state]
            values = []
            for action in range(n_actions):
                value = state_value_function(env,
                                             state, action, state_values, gamma)
                values.append(value)
            state_values[state] = np.max(values)
            delta = max(delta, np.abs(
                old_value - state_values[state]))
        if delta < threshold:
            break
    return state_values


def optimal_policy(env, state_values, gamma=0.999):
    actions = np.zeros(n_states, dtype=np.int8)
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


def play(env, policy, iterations=100):
    count = 0
    for iter in tqdm.tqdm(range(iterations)):
        obs = env.reset()
        while True:
            if count % 10 == 0:
                env.render()
            action = policy[int(obs)]
            obs, reward, done, _ = env.step(action)
            if done and reward == 1:
                count += 1
                print("Yay! you won!")
                break
            elif done and reward == 0:
                print("You! lost!")
                break

    print("You won!", count, "out of", iterations, "games.")


if __name__ == '__main__':
    random_gameplay(6)
    state_values = value_iteration(env)
    print(state_values.reshape((semi_states, semi_states)))
    policy = optimal_policy(env, state_values)
    print(policy.reshape((semi_states, semi_states)))
    play(env, policy)

    env.reset()
    state_values, policy = policy_iteration(env)
    print(state_values.reshape((semi_states, semi_states)))
    policy = optimal_policy(env, state_values)
    print(policy.reshape((semi_states, semi_states)))
    play(env, policy)
    
    env.close()

"""
4x4 gameplay

Action Values
[[0. 3. 0. 3.]
 [0. 0. 0. 0.]
 [3. 1. 0. 0.]
 [0. 2. 1. 0.]]
"""
