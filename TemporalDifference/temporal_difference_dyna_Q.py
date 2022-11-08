import numpy as np
import matplotlib.pyplot as plt
import tqdm
from collections import defaultdict
import gym
from mazelab.generators import random_maze
import colorama
import time
np.random.seed(0)

# 6,3


def color_sign(x):
    if x == 1:
        c = colorama.Fore.GREEN
    elif x == 0:
        c = colorama.Fore.RED
    else:
        c = colorama.Fore.BLUE
    return f'{c}{x}'


np.set_printoptions(formatter={'int': color_sign})


class EnvMaze:
    def __init__(self, env_id='RandomMaze-v0'):
        self._maze = random_maze(
            width=9, height=10, complexity=.75, density=.75)
        self.reset()

    def render(self, if_plot=False):
        for row in self.maze:
            print(row)
        print()

    def reset(self):
        self.maze = self._maze.copy()
        self.shapes = self.maze.shape
        self.movements = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        self.start = [1, 1]
        self.end = [7, 5]
        self.n_actions = len(self.movements)
        self.n_states = np.prod(self.maze.shape)
        self.current = self.start
        self.maze[self.current[0]][self.current[1]] = 8
        self.done = False
        return self.current[0]*self.maze.shape[1] + self.current[1]

    def step(self, action):
        if not (self.maze[self.current[0] + self.movements[action][0]][self.current[1] + self.movements[action][1]]):
            self.maze[self.current[0]][self.current[1]] = 0
            self.current[0] = np.clip(
                self.current[0] + self.movements[action][0], 1, self.shapes[0]-2)
            self.current[1] = np.clip(
                self.current[1] + self.movements[action][1], 1, self.shapes[1]-2)
        self.maze[self.current[0]][self.current[1]] = 8
        state = self.current[0]*self.maze.shape[1] + self.current[1]
        if self.current == self.end:
            reward = 10
            self.done = True
        else:
            reward = -1
        info = {}
        return state, reward, self.done, info, {}


class Model():
    def __init__(self, n_states, n_actions):
        self.transitions = np.zeros((n_states, n_actions), dtype=np.uint8)
        self.rewards = np.zeros((n_states, n_actions))

    def add(self, s, a, s_prime, r):
        # print(s, a, s_prime, r)
        self.transitions[s, a] = s_prime
        self.rewards[s, a] = r

    def sample(self):
        """ Return random state, action"""
        # Random visited state
        s = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])
        # Random action in that state
        a = np.random.choice(np.where(self.transitions[s] > 0)[0])

        return s, a

    def step(self, s, a):
        """ Return state_prime and reward for state-action pair"""
        s_prime = self.transitions[s, a]
        r = self.rewards[s, a]
        return s_prime, r


def dyna_Q_learning(env, iterations=1000):
    Q = defaultdict(lambda: np.zeros(env.n_actions, dtype=np.float32))
    model = Model(env.n_states, env.n_actions)
    policy = defaultdict(int)
    alpha = 0.01
    gamma = 0.95
    epsilon = 1
    decay = 1/iterations
    multi_step = 100
    minm_actions = np.inf
    maxm_actions = 0
    num_actions_per_ep = []
    avg_actions_per_ep = []
    minm_actions_per_ep = []
    maxm_actions_per_ep = []

    for iter in tqdm.tqdm(range(iterations)):
        old_state = env.reset()
        done = False
        actions_taken = 0
        while not done:
            if np.random.randn() <= epsilon:
                action = np.random.randint(0, env.n_actions)
            else:
                action = policy[old_state]
            state, reward, done, info, _ = env.step(action)

            Q[old_state][action] += alpha * \
                (reward + gamma*np.max([Q[state][a]
                                        for a in range(env.n_actions)]) - Q[old_state][action])
            model.add(old_state, action, state, reward)
            # env.render()
            multi_step = int(multi_step*0.95)
            for i in range(multi_step):
                s, a = model.sample()
                s_prime, r = model.step(s, a)
                Q[s][a] += alpha * (r + gamma*(np.max(Q[s_prime]) - Q[s][a]))
            policy[old_state] = np.argmax(
                [Q[old_state][a] for a in range(env.n_actions)])
            old_state = state
            actions_taken += 1
        num_actions_per_ep.append(actions_taken)
        avg_actions_per_ep.append(np.mean(num_actions_per_ep))
        minm_actions = min(minm_actions, actions_taken)
        maxm_actions = max(maxm_actions, actions_taken)
        minm_actions_per_ep.append(minm_actions)
        maxm_actions_per_ep.append(maxm_actions)
        epsilon -= decay

    plt.plot(num_actions_per_ep, label='Nums')
    plt.plot(avg_actions_per_ep, label='Avg')
    plt.plot(minm_actions_per_ep, label='Min')
    plt.plot(maxm_actions_per_ep, label='Max')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Num Actions')
    plt.show()
    return Q, policy


def games_trial(env, policy, no_of_games=100):
    count = 0
    for games in tqdm.tqdm(range(no_of_games)):
        done = False
        state = env.reset()
        while not done:
            action = policy[state]
            state, reward, done, info, _ = env.step(action)

            if reward == 10:
                count += 1
            print(action)
            env.render()
            time.sleep(0.15)

    print(" No of games won:", count, "out of", no_of_games)


if __name__ == '__main__':
    env = EnvMaze()
    print(env.reset())
    env.render()
    Q, policy = dyna_Q_learning(env)
    env.reset()
    env.render()
    games_trial(env, policy, no_of_games=100)
