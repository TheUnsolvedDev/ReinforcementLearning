import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from params import *
from model import *
from replay_buffer import *
from DQN_agent import *


class DoubleDeepQNetwork_agent(DeepQNetwork_agent):
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        super().__init__(epsilon, gamma, memory_capacity, sample_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape1:
            q_values = self.cnn(states)
            next_q_values = tf.stop_gradient(self.target_cnn(next_states))
            q_next_values = self.cnn(next_states)

            q_expected = tf.reduce_sum(
                q_values * tf.one_hot(actions, 6), axis=1)
            next_actions = tf.argmax(q_next_values, axis=1)
            q_targets = tf.reduce_sum(
                next_q_values*tf.one_hot(actions, 6), axis=1)

            rewards = tf.cast(rewards, tf.float32)
            q_targets = tf.add(rewards, (self.gamma)*q_targets*(1.-dones))
            losses = self.loss(q_targets, q_expected)

        grads = tape1.gradient(losses, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables))

    def save_model(self):
        self.cnn.save_weights("DDQN_model.h5")
        self.target_cnn.save_weights("DDQN_target_model.h5")

    def load_model(self):
        self.cnn.load_weights("DDQN_target_model.h5")
        self.target_cnn.load_weights("DDQN_target_model.h5")
