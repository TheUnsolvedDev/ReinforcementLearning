import gym
import numpy as np
from kaggle_environments import evaluate, make, utils

from params import *


class ConnectX(gym.Env):
    def __init__(self):
        self.env = make('connectx', debug=False)
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(
            config.columns * config.rows)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def get_state(self):
        return self.env.state

    def game_over(self):
        return self.env.done

    def current_player(self):
        active = -1
        if self.env.state[0].status == "ACTIVE":
            active = 0
        if self.env.state[1].status == "ACTIVE":
            active = 1
        return active

    def get_configuration(self):
        return self.env.configuration


if __name__ == '__main__':
    obj = ConnectX()
