import numpy as np
import tensorflow as tf
import gc

from params import *


class ReplayBuffer:
    def __init__(self, capacity, batch_size=128):
        self.capacity = capacity
        self.batch_size = batch_size
        self.mem_cntr = 0
        input_shape = list(INPUT_SHAPE)
        self.states = np.zeros((self.capacity, *input_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int8)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, *input_shape), dtype=np.float32)

    def add(self, state, reward, action, next_state, done):
        index = self.mem_cntr % self.capacity
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.next_states[index] = next_state
        self.mem_cntr += 1

    def sample(self):
        max_mem = min(self.capacity, self.mem_cntr)
        batch = np.random.choice(max_mem, self.batch_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        dones = self.dones[batch]
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return self.mem_cntr
