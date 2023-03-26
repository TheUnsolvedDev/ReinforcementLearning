import numpy as np
import tensorflow as tf
<<<<<<< Updated upstream
<<<<<<< Updated upstream
import gc
=======
import random
>>>>>>> Stashed changes
=======
import random
>>>>>>> Stashed changes

from params import *


class ReplayBuffer:
    def __init__(self, capacity, batch_size=128):
        self.capacity = capacity
        self.batch_size = batch_size

    def add(self, state, reward, action, next_state, done):
        experience = (state, reward, action, next_state, done)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        self.memory.append(experience)

    def sample(self):
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []

        batch = random.sample(self.memory, self.batch_size)
        for experience in batch:
            states.append(experience[0])
            rewards.append(experience[1])
            actions.append(experience[2])
            next_states.append(experience[3])
            dones.append(experience[4])

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int8)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
