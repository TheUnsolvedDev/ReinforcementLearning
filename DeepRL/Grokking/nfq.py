import gymnasium as gym
from gymnasium import wrappers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import subprocess
import os.path
import tempfile
import random
import base64
import pprint
import glob
import time
import json
import sys
import io
import os
import gc

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MAX_EPISODES = 1000
MAX_STEPS = 10000
env = gym.make('CartPole-v1', max_episode_steps=500, render_mode='human')

gamma = 0.99
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

LEAVE_PRINT_EVERY_N_SECS = 60
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
def BEEP(): return os.system("printf '\a'")


RESULTS_DIR = os.path.join('.', 'results')


def FA(in_dim, out_dim, out_activation='linear'):
    inputs = tf.keras.layers.Input(in_dim)
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(out_dim, activation=out_activation)(x)
    return tf.keras.Model(inputs, outputs)


class GreedyStrat():
    def __init__(self) -> None:
        self.exploratory_action_take = False

    def select_action(self, model, state):
        q_values = model([state], training=False).numpy().squeeze()
        return np.argmax(q_values)


class EGreedyStrat():
    def __init__(self, epsilon=0.1) -> None:
        self.epsilon = epsilon
        self.exploratory_action_take = False

    def select_action(self, model, state):
        self.exploratory_action_take = False
        q_values = model([state], training=False).numpy().squeeze()
        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(q_values))
        self.exploratory_action_take = action != np.argmax(q_values)
        return action


class NFQ():
    def __init__(self, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn, evaluation_strategy_fn, batch_size, epochs) -> None:
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.batch_size = batch_size
        self.epochs = epochs

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(dones)


if __name__ == '__main__':
    pass
