import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import tqdm


class CliffHanger:
    def __init__(self, size=6):
        self.state = np.zeros((size, size))
        self.size = size
        self.start = [size//2, 0]
        self.end = [size//2, size//2]
        self.state[self.start[0]][self.start[1]] = 1
        self.state[self.end[0]][self.end[1]] = 10
        self.n_states = (self.size**2)
        self.n_actions = 4
        self.current = self.start
        self.action_map = {
            0: [1, 0],
            1: [0, 1],
            2: [-1, 0],
            3: [0, -1]
        }
        self.done = False
        self.counter = 0
        self.max_states = 10000

    def reset(self):
        self.counter = 0
        self.state = np.zeros((self.size, self.size))
        self.start = [self.size//2, 0]
        self.end = [self.size//2, self.size//2]
        self.state[self.start[0]][self.start[1]] = 1
        self.state[self.end[0]][self.end[1]] = 10
        self.current = self.start
        return self.current[0]*6 + self.current[1]

    def sample(self):
        return np.random.randint(0, 4)

    def render(self):
        print(self.state, self.counter)
        print()

    def step(self, action, step_show=False):
        self.counter += 1
        if not step_show:
            self.state[self.current[0]][self.current[1]
                                        ] = 0 if self.current != self.end else 10
        self.current[0] = np.clip(
            self.current[0]+self.action_map[action][0], 0, 5)
        self.current[1] = np.clip(
            self.current[1]+self.action_map[action][1], 0, 5)

        if self.current[1] == self.size//2:
            self.current[0] = np.clip(self.current[0]-2, 0, 5)
        elif self.current[1] == self.size//2 - 1 or self.current[1] == self.size//2 + 1:
            self.current[0] = np.clip(self.current[0]-1, 0, 5)

        self.state[self.current[0]][self.current[1]] = 1
        if self.current == self.end:
            reward = 100
            self.done = True
        else:
            if self.counter >= self.max_states:
                self.done = True
            reward = -1

        return self.current[0]*6 + self.current[1], reward, self.done, {}


def Q_learning(env, iterations=100000):
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
    policy = defaultdict(int)
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1
    decay = 1/iterations

    for iter in tqdm.tqdm(range(iterations)):
        old_state = env.reset()
        done = False
        while not done:
            if np.random.randn() <= epsilon:
                action = env.sample()
            else:
                action = policy[old_state]
            state, reward, done, info = env.step(action)

            Q[old_state][action] += alpha * \
                (reward + gamma*np.max([Q[state][a]
                                        for a in range(n_actions)]) - Q[old_state][action])
            policy[old_state] = np.argmax(
                [Q[old_state][a] for a in range(n_actions)])
            old_state = state
        # epsilon -= decay

    return Q, policy


if __name__ == '__main__':
    level = CliffHanger()
    n_actions = level.n_actions
    n_states = level.n_states
    state = level.reset()
    done = level.done

    Q, policy = Q_learning(level)
    map_arrow_set = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←'
    }
    for i in range(6):
        array = []
        for j in range(6):
            array.append(map_arrow_set[policy[i*6+j]])
        print(array)

    level = CliffHanger()
    done = level.done
    state = level.reset()
    while not done:
    	action = policy[state]
    	level.render()
    	state, reward, done, info = level.step(action)
    	time.sleep(0.2)
    	print(map_arrow_set[action])
