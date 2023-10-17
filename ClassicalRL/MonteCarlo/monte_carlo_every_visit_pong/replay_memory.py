import numpy as np
from collections import deque
import random

from params import *


class MemoryBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buffer_size = size
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = BATCH_SIZE

    def push(self, state, values):
        self.memory.append((state, values))

    def sample(self):
        size = min(self.batch_size, self.length())
        experiences = random.sample(self.memory, k=size)

        states = []
        values = []
        for now in experiences:
            if now[0] is not None and now[1] is not None:
                states.append(now[0])
                values.append(now[1])
            else:
                break

        return (np.array(states), np.array(values))

    def length(self):
        return len(self.memory)
