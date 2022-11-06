import numpy as np
import matplotlib.pyplot as plt
import time


class CliffHanger:
    def __init__(self, size=6):
        self.state = np.zeros((size, size))
        self.size = size
        self.start = [size//2, 0]
        self.end = [size//2, size//2]
        self.state[self.start[0]][self.start[1]] = 1
        self.state[self.end[0]][self.end[1]] = 10
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

    def render(self):
        print(self.state,self.counter)
        print()

    def step(self, action):
        self.counter += 1
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
            reward = 10
            self.done = True
        else:
            if self.counter >= self.max_states:
                self.done = True
            reward = -1

        return self.current[0]*6 + self.current[1], reward, self.done, {}


if __name__ == '__main__':
    level = CliffHanger()
    state = level.reset()
    done = level.done
    while not done:
        level.render()
        action = np.random.randint(0, 4)
        state, reward, done, info = level.step(action)
