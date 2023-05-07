import tensorflow as tf
import numpy as np

GAMMA = 0.99


class PrioritizedReplayBuffer:
    def __init__(self, max_size=10_000, alpha=0.6, beta=0.4):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.epsilon = 1e-6

    def put(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if self.size < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        if self.size == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*samples))
        states = np.array(states, dtype=np.float32).reshape(
            batch_size, -1)
        next_states = np.array(next_states, dtype=np.float32).reshape(
            batch_size, -1)
        done = np.array(done, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        return states, actions, rewards, next_states, done, samples, indices, weights

    def calculate_new_priorities(self, experiences, q_network, target_network):
        new_priorities = []
        for state, action, reward, next_state, done in experiences:
            # Calculate the TD error
            q_values = q_network.predict(state)
            target = q_network.predict(next_state) if not done else 0.0
            target = reward + GAMMA * \
                np.max(target_network.predict(next_state)) * (1 - done)
            td_error = abs(target - q_values[0][action])
            new_priorities.append(td_error)
        return new_priorities

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
