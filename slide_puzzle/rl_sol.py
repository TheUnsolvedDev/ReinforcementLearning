import numpy as np
import tensorflow as tf
import time
import tqdm
import pickle
import sys
np.set_printoptions(threshold=sys.maxsize)

from puzzle import NPuzzle

env = NPuzzle(3)
env.reset()

n_actions = 4
n_states = 181440

def state_value_function(env, state, action, state_values, gamma=0.9):
    return np.sum([prob*(reward+gamma*state_values[env.sti[tuple(next_state)]]) for
                   prob, next_state, reward, done in [env.P[env.its[state]][action]]])
    
def policy_evaluation(env, state_values, policy, iterations=500, threshold=0.001, gamma=0.999):
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
    with open('state_values.pkl', 'wb') as f:
        pickle.dump(state_values, f)
    return state_values, policy, delta


def policy_improvement(env, state_values, policy, gamma):
    for state in range(n_states):
        policy[state] = np.argmax([state_value_function(env,
                                                        state, action, state_values, gamma) for action in range(n_actions)])
    with open('policy.pkl','wb') as f:
        pickle.dump(policy, f)
    return state_values, policy


def policy_iteration(env, gamma=0.95):
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
        done = False
        while not done:
            action = int(policy[env.sti[tuple(obs)]])
            obs, reward, done, _ = env.step(action)
            if done and reward == 20:
                count += 1
                print("Yay! you won!")
            elif done and reward != 20:
                print("You! lost!")
            
            if count % 10 == 0:
                time.sleep(0.2)
                env.render('human')

    print("You won!", count, "out of", iterations, "games.")

if __name__ == '__main__':
    done = False
    count = 0
    
    state_values,policy = policy_iteration(env)
    with open('state_values.pkl','rb') as state_values:
        state_values = pickle.load(state_values)
            
    with open('policy.pkl','rb') as policy:
        policy = pickle.load(policy)
    
    # print(state_values,policy)
    
    play(env,policy,iterations = 10000)
    
