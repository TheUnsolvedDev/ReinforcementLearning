import numpy as np
import gym
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env2 = gym.make('CartPole-v1', render_mode='human')
alpha = 0.2
gamma = 0.99
n_actions = 2

poleThetaSpace = np.linspace(-0.209, 0.209, 10)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)


def get_state(observation):
    cartX, cartXdot, cartTheta, cartThetaDot = observation
    cartX = int(np.digitize(cartX, cartPosSpace))
    cartXdot = int(np.digitize(cartXdot, cartVelSpace))
    cartTheta = int(np.digitize(cartTheta, poleThetaSpace))
    cartThetaDot = int(np.digitize(cartThetaDot, poleThetaVelSpace))
    return (cartX, cartXdot, cartTheta, cartThetaDot)


def n_step_sarsa(env, iterations=10000, n=8):
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    policy = defaultdict(int)
    decay = 1/iterations
    epsilon = 1.0

    state_memory = np.zeros((n, 4))
    action_memory = np.zeros(n)
    reward_memory = np.zeros(n)
    scores = []

    for iter in tqdm.tqdm(range(iterations)):
        old_state = env.reset()[0]
        old_state = get_state(old_state)
        done = False
        t = 0
        score = 0
        T = np.inf
        if np.random.randn() <= epsilon:
            old_action = env.action_space.sample()
        else:
            old_action = policy[old_state]
        action_memory[t % n] = old_action
        state_memory[t % n] = old_state
        while not done:
            state, reward, done, info, _ = env.step(old_action)
            score += reward
            state = get_state(state)
            state_memory[(t+1) % n] = state
            reward_memory[(t+1) % n] = reward
            if done:
                T = t+1

            if np.random.randn() <= epsilon:
                action = env.action_space.sample()
            else:
                action = policy[old_state]
            action_memory[(t+1) % n] = action
            tau = t - n + 1
            if tau >= 0:
                G = [gamma**(j-tau - 1)*reward_memory[j % n]
                     for j in range(tau + 1, min(tau+n, T)+1)]
                G = np.sum(G)
                if tau + n < T:
                    s = get_state(state_memory[(t+n) % n])
                    a = int(action_memory[(tau + n) % n])
                    G += gamma**n*Q[s][a]
                s = get_state(state_memory[tau % n])
                a = int(action_memory[tau % n])
                Q[s][a] += alpha*(G-Q[s][a])
            # print('tau ', tau, '| Q %.2f' % \
            #        Q[get_state(state_memory[tau%n])][int(action_memory[tau%n])])
            t += 1

            policy[old_state] = np.argmax(Q[old_state])
            old_state = state
            old_action = action

        for tau in range(t-n+1, T):
            G = [gamma**(j-tau - 1)*reward_memory[j % n]
                 for j in range(tau + 1, min(tau+n, T)+1)]
            G = np.sum(G)
            if tau + n < T:
                s = get_state(state_memory[(t+n) % n])
                a = int(action_memory[(tau + n) % n])
                G += gamma**n*Q[s][a]
            s = get_state(state_memory[tau % n])
            a = int(action_memory[tau % n])
            Q[s][a] += alpha*(G-Q[s][a])
        epsilon -= decay
        scores.append(score)
    plt.plot(scores)
    plt.show()
    return Q, policy


def games_trial(env, policy, no_of_games=1000):
    count = 0
    for games in tqdm.tqdm(range(no_of_games)):
        done = False
        state = env.reset()[0]
        state = tuple(get_state(state))
        while not done:
            action = policy[state]
            state, reward, done, info, _ = env.step(action)
            state = tuple(get_state(state))

            if reward == 1:
                count += 1

    print(" No of games won:", count, "out of", no_of_games)


if __name__ == '__main__':
    Q, policy = n_step_sarsa(env)
    # print(policy)

    # V = np.array([np.max([Q[state][action] for action in range(n_actions)]) for state in range(n_states)])
    # print(V.reshape(semi_states,semi_states))
    games_trial(env2, policy, no_of_games=100)
    env.close()
    env2.close()
