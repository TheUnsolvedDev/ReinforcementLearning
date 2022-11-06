import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import tqdm


class CliffHanger:
    def __init__(self, size=10):
        self.state = np.zeros((size, size))
        self.size = size
        self.start = [3, 0]
        self.end = [3, 7]
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
        self.wind1 = [3, 4, 5, 8]
        self.wind2 = [6, 7]
        # self.counter = 0
        # self.max_states = 1000

    def reset(self):
        self.counter = 0
        self.state = np.zeros((self.size, self.size))
        self.start = [3, 0]
        self.end = [3, 7]
        self.done = False
        self.state[self.start[0]][self.start[1]] = 1
        self.state[self.end[0]][self.end[1]] = 10
        self.current = self.start
        self.wind1 = [3, 4, 5, 8]
        self.wind2 = [6, 7]
        return self.current[0]*self.size + self.current[1]

    def sample(self):
        return np.random.randint(0, 4)

    def render(self):
        print(self.state, self.counter)
        print()

    def step(self, action, step_show=False):
        # self.counter += 1
        if not step_show:
            self.state[self.current[0]][self.current[1]
                                        ] = 0 if self.current != self.end else 10
        self.current[0] = np.clip(
            self.current[0]+self.action_map[action][0], 0, self.size - 1)
        self.current[1] = np.clip(
            self.current[1]+self.action_map[action][1], 0, self.size - 1)

        if self.current[1] in self.wind1:
            self.current[0] = np.clip(self.current[0]-1, 0, self.size - 1)
        if self.current[1] in self.wind2:
            self.current[0] = np.clip(self.current[0]-2, 0, self.size - 1)

        self.state[self.current[0]][self.current[1]] = 1
        if self.current == self.end:
            reward = 10
            self.done = True
        else:
            # if self.counter >= self.max_states:
            #     self.done = True
            reward = -1

        return self.current[0]*self.size + self.current[1], reward, self.done, {}


def Q_learning(env, iterations=500):
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    policy = defaultdict(int)
    alpha = 0.2
    gamma = 1
    epsilon = 0.05
    decay = 1/iterations

    for iter in tqdm.tqdm(range(iterations)):
        trajectry = []
        old_state = env.reset()
        done = False
        while not done:
            trajectry.append(old_state)
            if np.random.randn() <= epsilon:
                action = env.sample()
            else:
                action = policy[old_state]
            state, reward, done, info = env.step(action)
            action_best = np.argmax(Q[state])
            Q[old_state][action] += alpha * \
                (reward + gamma*Q[state][action_best] - Q[old_state][action])
            policy[old_state] = action_best
            old_state = state
        # epsilon -= decay

    return Q, policy, trajectry


def trajectoryPath(traj):
    # Initialize gridworld
    world_map = np.zeros((10, 10))
    for i, state in enumerate(traj):
        x = int(state % 10)
        y = int((state - x) / 10)
        world_map[y, x] = i + 1
    print(world_map)
    print("\n")


if __name__ == '__main__':
    level = CliffHanger()
    n_actions = level.n_actions
    n_states = level.n_states
    state = level.reset()
    done = level.done

    Q, policy, trajectry = Q_learning(level)
    map_arrow_set = {
        0: '↓',
        1: '→',
        2: '↑',
        3: '←'
    }
    for i in range(level.size):
        array = []
        for j in range(10):
            array.append(map_arrow_set[policy[i*level.size+j]])
        print(array)

    trajectoryPath(trajectry)

    # level = CliffHanger()
    # done = level.done
    # state = level.reset()
    # while not done:
    #     action = policy[state]
    #     level.render()
    #     state, reward, done, info = level.step(action)
    #     time.sleep(0.2)
    #     print(map_arrow_set[action])
