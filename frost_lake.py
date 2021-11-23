import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import copy
import seaborn
from tqdm import tqdm

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

actions = {LEFT: '<', RIGHT: '>', UP: '^', DOWN: 'V'}

maze = [
    ["S", "F", "F", "F"],  # [0,1,2,3]
    ["F", "H", "F", "H"],  # [4,5,6,7]
    ["F", "F", "F", "H"],  # [8,9,10,11]
    ["H", "F", "F", "G"]  # [12,13,14,15]
]


class Environment:
    def __init__(self, maze) -> None:
        self._maze = np.array(maze)
        self.nrow, self.ncol = len(self._maze), len(self._maze[0])

        self.target = (self.nrow - 1, self.ncol - 1)
        self.free_cells = [(i, j) for i in range(self.nrow)
                           for j in range(self.ncol) if self._maze[i, j] == 'F']
        self.reward = (-3, -0.5, 100)

        self.nactions = 4
        self.nstates = self.nrow * self.ncol
        self.reset(rat=(0, 0))

        num_state = self.nstates
        num_action = self.nactions

        trans_prob = np.zeros((num_state, num_action, num_state))
        reward = np.zeros((num_state, num_action, num_state))

        counter = 0
        while counter < 5000:
            state = self.reset()
            done = False

            while not done:
                random_action = np.random.randint(0, 4)
                new_state, r, done = self.step(random_action)
                trans_prob[state][random_action][new_state] += 1
                reward[state][random_action][new_state] += r

                state = new_state
                counter += 1

        # normalization
        for i in range(trans_prob.shape[0]):
            for j in range(trans_prob.shape[1]):
                norm_coeff = np.sum(trans_prob[i, j, :])
                norm_reward = np.sum(reward[i,j,:])
                if norm_coeff:
                    trans_prob[i, j, :] /= norm_coeff
                    # reward[i,j, :] /= norm_reward


        self.P = trans_prob
        self.reward = reward

    def get_reward(self, state, action, state_prime):
        return self.reward[state][action][state_prime]

    def reset(self, rat=(0, 0)) -> int:
        self.rat = rat
        self.maze = np.copy(self._maze)

        nrow, ncol = self.maze.shape
        rat_row, rat_col = rat

        self.state = 4*rat_row + rat_col

        self.total_reward = 0
        return self.state

    def location_to_state(self, location: tuple) -> int:
        nrow, ncol = self.maze.shape
        row, col = location
        return 4*row + col

    def state_to_location(self, state: int) -> tuple:
        nrow, ncol = self.maze.shape
        row = state // 4
        col = state % 4
        return (row, col)

    def increment(self, row, col, action):
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)

        return (row, col)

    def update_transition(self, row, col, action):
        new_row, new_col = self.increment(row, col, action)
        new_state = self.location_to_state((new_row, new_col))
        new_letter = self.maze[new_row, new_col]
        done = new_letter == 'G' or new_letter == 'H'
        if self.maze[self.rat] == 'G':
            reward = self.reward[2]
        elif self.maze[self.rat] == 'F' or 'S':
            reward = self.reward[1]
        elif self.maze[self.rat] == 'H':
            reward = self.reward[0]

        return (new_state, reward, done)

    def render(self) -> None:
        nrow, ncol = self.maze.shape
        maze = np.copy(self.maze)
        maze[self.state // 4, self.state % 4] = 'R'
        print(maze)

    def valid_actions(self, location: tuple) -> list:
        row, col = location
        actions = [LEFT, RIGHT, UP, DOWN]
        nrow, ncol = self.maze.shape
        if row == 0:
            actions.remove(UP)
        if row == nrow - 1:
            actions.remove(DOWN)
        if col == 0:
            actions.remove(LEFT)
        if col == ncol - 1:
            actions.remove(RIGHT)
        return actions

    def step(self, action: int, debug_actions=False, done_debug=False) -> tuple:
        nrow, ncol = self.maze.shape
        rat_row, rat_col = self.rat

        valid_actions = self.valid_actions((rat_row, rat_col))
        if action not in valid_actions:
            return self.state, 0, False
        else:
            self.rat = self.increment(rat_row, rat_col, action)
            self.state = self.location_to_state(self.rat)
        self.maze[rat_row, rat_col] = 'S' if (
            rat_row, rat_col) == (0, 0) else 'F'

        if self.maze[self.rat] == 'G':
            reward = self.reward[2]
        elif self.maze[self.rat] == 'F' or 'S':
            reward = self.reward[1]
        elif self.maze[self.rat] == 'H':
            reward = self.reward[0]
        done = self.maze[self.rat] == 'G' or self.maze[self.rat] == 'H'
        if debug_actions:
            print(actions[action])
        if done:
            self.reset()
            if done_debug:
                print('Episode finished')

        return self.state, reward, done


def policy_evaluation(env, policy, value, reward, gamma=0.8, iteration=5):
    num_state = env.nstates
    for i in range(iteration):
        for state in range(num_state):
            val = 0
            for state_prime in range(num_state):
                val += env.P[state][policy[state]][state_prime] * \
                    (env.get_reward(
                        state, policy[state], state_prime) + gamma * value[state_prime])
            value[state] = val

    return value


def policy_improve(env, policy, value, gamma=0.8):
    num_state = env.nstates
    policy_stable = True
    best_action = np.random.randint(0, 4)
    for state in range(num_state):
        max_val = -1
        old_action = policy[state]
        for action in range(env.nactions):
            val = 0
            for state_prime in range(num_state):
                val += env.P[state][action][state_prime] * \
                    (env.get_reward(state, action, state_prime) +
                     gamma * value[state_prime])
            if val > max_val:
                max_val = val
                best_action = action
        policy[state] = best_action
    return policy


def policy_iteration(env, gamma=0.8, iteration=50):
    num_state = env.nstates
    policy = np.zeros(num_state, dtype=int)
    value = np.zeros(num_state)
    for i in tqdm(range(iteration)):
        value = policy_evaluation(env, policy, value, gamma)
        policy = policy_improve(env, policy, value, gamma)

    return policy, value


if __name__ == '__main__':
    env = Environment(maze)
    env.render()

    # # print(optimal_value_function(env))
    # # print(policy_iteration(env))

    PI_policy, value = policy_iteration(env, iteration=100)
    PI_policy = np.array(list(map(lambda x: actions[x], PI_policy)))
    print()
    print(PI_policy.reshape(env.nrow, env.ncol))

    print(value.reshape(env.nrow, env.ncol))
