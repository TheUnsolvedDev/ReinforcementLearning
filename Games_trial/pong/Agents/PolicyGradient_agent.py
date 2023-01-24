import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from params import *
from model import *
from replay_buffer import *


class Reinforce_agent:
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE, use_baseline=False):
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.sample_size = sample_size
        self.use_baseline = use_baseline

        self.cnn = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.optimizer = tf.keras.optimizers.Adam(ALPHA)
        self.memory = ReplayBuffer(BUFFER_SIZE)

        if self.use_baseline:
            self.baseline_cnn = cnn(INPUT_SHAPE, 1)
            self.baseline_optimizer = tf.keras.optimizers.Adam(ALPHA)

        self.states = []
        self.rewards = []
        self.actions = []
        self.masks = []

    def reset_memory(self):
        del self.states[:]
        del self.rewards[:]
        del self.actions[:]
        del self.masks[:]

    def compute_returns(self, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns

    def step(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(1 - done)
        if done:
            # print('Starting to learn...')
            self.learn()

    def act(self, state, eps):
        state = tf.expand_dims(state, axis=0)
        values = self.cnn.predict(state, verbose=False)
        dist = tfp.distributions.Categorical(logits=values)
        action = dist.sample()
        return int(action.numpy()[0])

    def get_advantage(self, returns, observations):
        values = self.baseline_cnn(observations)
        advantages = returns - values
        advantages = (advantages-np.mean(advantages)) / \
            np.sqrt(np.sum(advantages**2))
        return advantages

    @tf.function
    def advantage_loss(self, returns, predictions):
        return tf.keras.losses.mean_squared_error(
            y_true=returns, y_pred=predictions)

    @tf.function
    def get_loss(self, values, actions, returns):
        dist = tfp.distributions.Categorical(logits=values)
        log_prob = dist.log_prob(actions)
        loss = -tf.reduce_mean(log_prob * returns)
        return loss

    def learn(self):
        returns = tf.convert_to_tensor(
            self.compute_returns(0), dtype=tf.float32)
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)

        if self.use_baseline:
            advantages = self.get_advantage(returns, states)
            with tf.GradientTape() as tape:
                predictions = self.baseline_cnn(states)
                loss = self.advantage_loss(returns, predictions)
            grads = tape.gradient(loss, self.baseline_cnn.trainable_weights)
            self.baseline_optimizer.apply_gradients(
                zip(grads, self.baseline_cnn.trainable_weights))
            returns = advantages

        with tf.GradientTape() as tape:
            values = self.cnn(states)
            loss = self.get_loss(values, actions, returns)

        grads = tape.gradient(loss, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables))

        self.reset_memory()

    def save_model(self):
        if self.use_baseline:
            self.cnn.save_weights("PG_baseline_model.h5")
        else:
            self.cnn.save_weights("PG_model.h5")

    def load_model(self):
        if self.use_baseline:
            self.cnn.load_weights("PG_baseline_model.h5")
        else:
            self.cnn.load_weights("PG_model.h5")
