import os
import numpy as np
from tabulate import tabulate


class BaseGridworld:
    """
    Defines the base class for the Gridworld MDP in Chapters 3, 4, 6, 7.

    State representation: (x,y) tuple; width and height coordinate standard cartisian.
    Action representation: (x_offset, y_offset) tuple; where offset is distance in the x or y direction.

    E.g. (0, 0) is center; x increases in right direction, y increases in up direction.
         Action (1, 1) is increments x with +1 and y with +1; Thus state (0,0) + action (1,1) = next_state (1,1).

    """

    def __init__(self, width, height, start_state=None, goal_state=None, terminal_states=[], blocked_states=[]):
        """
        Args
            width, height -- ints; dimensions of the grid for x and y.
            start_state -- tuple; agent start state.
            goal_state -- tuple; agent goal state.
            terminal_states -- list of tuples; special states (if any) with specified reward and action (e.g. cliff in Ch 6)
            blocked_states -- list of tuples; special states which are not accessible to the agent (e.g. walls in Ch 8)
        """
        self.width = width
        self.height = height
        self.start_state = start_state
        self.goal_state = goal_state
        self.terminal_states = terminal_states
        self.blocked_states = blocked_states

        self.reset_state()

    def get_possible_actions(self, state):
        all_actions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        return all_actions

    def get_states(self):
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def get_reward(self, state, action, next_state):
        raise NotImplementedError

    def get_state_reward_transition(self, state, action):
        next_state = np.array(state) + np.array(action)
        next_state = self._clip_state_to_grid(next_state)
        if self.is_blocked(next_state):
            next_state = state
        next_state = int(next_state[0]), int(next_state[1])
        reward = self.get_reward(state, action, next_state)
        return next_state, reward

    def _clip_state_to_grid(self, state):
        x, y = state
        return np.clip(x, 0, self.width-1), np.clip(y, 0, self.height-1)

    def is_goal(self, state):
        return tuple(state) == self.goal_state

    def is_terminal(self, state):
        return tuple(state) in self.terminal_states

    def is_blocked(self, state):
        return tuple(state) in self.blocked_states

    def reset_state(self):
        self.state = self.start_state
        return self.state


def action_to_nwse(action):
    """ translate an action from tuple (e.g. (1,0)) to letter coordinates (e.g. 'e') """
    x, y = action
    ret = ''
    if y == +1:
        ret += 'n'
    if y == -1:
        ret += 's'
    if x == +1:
        ret += 'e'
    if x == -1:
        ret += 'w'
    return ret


def print_grid(mdp, f=None):
    grid = [[' ' for x in range(mdp.width)] for y in range(mdp.height)]
    for (x, y) in mdp.get_states():
        marker = ' '
        if (x, y) == mdp.start_state:
            marker = 'S'
        if (x, y) == mdp.goal_state:
            marker = 'G'
        if mdp.is_terminal((x, y)):
            marker = 'T'
        if mdp.is_blocked((x, y)):
            marker = 'B'

        grid[y][x] = marker

    grid = grid[::-1]

    print(tabulate(grid, tablefmt='grid'), file=f)
    return grid


def print_path(mdp, states_visited, f=None):
    grid = print_grid(mdp, f=open(os.devnull, 'w'))
    grid = grid[::-1]

    for (x, y) in states_visited:
        skip_states = ['S', 'G']
        if grid[y][x] not in skip_states:
            grid[y][x] = 'o'

    grid = grid[::-1]

    print(tabulate(grid, tablefmt='grid'), file=f)
    return grid
