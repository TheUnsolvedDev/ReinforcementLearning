import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size=128):
        self.capacity = capacity
        self.memory = []
        self.batch_size = batch_size

    def add(self, state, reward, action, next_state, done):
        mems = (state, action, reward, next_state, done)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.memory.append(mems)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []

        for state, reward, action, next_state, done in experiences:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

        states = np.array(states, np.float32)
        actions = np.array(actions, np.uint8)
        rewards = np.array(rewards, np.float32)
        next_states = np.array(next_states, np.float32)
        dones = np.array(dones, np.float32)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
