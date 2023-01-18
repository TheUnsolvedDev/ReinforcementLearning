import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size=128):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])

    def add(self, state, reward, action, next_state, done):
        mems = self.experience(state, action, reward, next_state, done)
        self.memory.append(mems)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor(
            np.array([e.state for e in experiences if e is not None]))
        actions = tf.convert_to_tensor(
            np.array([e.action for e in experiences if e is not None]))
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in experiences if e is not None]))
        next_states = tf.convert_to_tensor(
            np.array([e.next_state for e in experiences if e is not None]))
        dones = tf.convert_to_tensor(
            np.array([e.done for e in experiences if e is not None]).astype(np.float32))
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
