import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from agents.params import *
from agents.models import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def discounted_rewards(rewards, gamma=GAMMA):
    discounted = np.zeros_like(rewards)
    running_sum = 0
    for i in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[i]
        discounted[i] = running_sum
    return discounted


def calculate_baselines(rewards):
    baselines = np.zeros_like(rewards)
    running_sum = 0
    for i in reversed(range(len(rewards))):
        running_sum = running_sum + rewards[i]
        baselines[i] = running_sum / (len(rewards) - i)
    return baselines

