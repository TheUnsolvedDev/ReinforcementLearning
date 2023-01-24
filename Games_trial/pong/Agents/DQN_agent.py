import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from params import *
from model import *
from replay_buffer import *

class DeepQNetwork_agent:
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.sample_size = sample_size

        self.cnn = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.target_cnn = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(ALPHA)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0
        self.update_every_step = 1
        self.replay_after = 512
        self.tau = TAU

    def step(self, state, reward, action, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every_step

        if not self.t_step:
            if len(self.memory) >= self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0):
        if np.random.rand() < eps:
            return np.random.randint(0, ACTION_SIZE)
        else:
            state = tf.expand_dims(state, axis=0)
            return np.argmax(self.cnn.predict(state, verbose=False))

    @tf.function
    def get_loss(self, q_expected, rewards, q_targets, dones):
        rewards = tf.cast(rewards, tf.float32)
        q_targets = tf.add(rewards, (self.gamma)*q_targets*(1.-dones))
        losses = self.loss(q_targets, q_expected)
        return losses

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape1:
            q_values = self.cnn(states)
            next_q_values = tf.stop_gradient(self.target_cnn(next_states))
            q_expected = tf.math.reduce_sum(
                q_values * tf.one_hot(actions, 6), axis=1)
            q_targets = tf.math.reduce_max(next_q_values, axis=1)
            losses = self.get_loss(
                q_expected, rewards, q_targets, dones)

        grads = tape1.gradient(losses, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables))

        self.soft_update()

    def soft_update(self):
        for target, policy in zip(self.target_cnn.layers, self.cnn.layers):
            if isinstance(target, tf.keras.layers.Conv2D) or isinstance(target, tf.keras.layers.Dense):
                target_weight = target.get_weights()[0]
                target_bias = target.get_weights()[1]
                policy_weight = policy.get_weights()[0]
                policy_bias = policy.get_weights()[1]

                update_weights = self.tau*policy_weight + \
                    (1-self.tau)*target_weight
                update_bias = self.tau*policy_bias + \
                    (1 - self.tau)*target_bias

                target.set_weights([update_weights, update_bias])

    def save_model(self):
        self.cnn.save_weights("DQN_model.h5")
        self.target_cnn.save_weights("DQN_target_model.h5")

    def load_model(self):
        self.cnn.load_weights("DQN_target_model.h5")
        self.target_cnn.load_weights("DQN_target_model.h5")