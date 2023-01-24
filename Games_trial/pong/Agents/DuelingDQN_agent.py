import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from params import *
from model import *
from replay_buffer import *
from DQN_agent import *

class DuelingDQN_agent(DeepQNetwork_agent):
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        super().__init__(epsilon, gamma, memory_capacity, sample_size)