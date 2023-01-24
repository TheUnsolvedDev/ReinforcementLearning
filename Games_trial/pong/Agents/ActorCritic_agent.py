import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from params import *
from model import *
from replay_buffer import *


class ActorCritic_agent:
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.sample_size = sample_size

        self.actor_cnn = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.critic_cnn = cnn(INPUT_SHAPE, 1)
        self.actor_optimizer = tf.keras.optimizers.Adam(ALPHA)
        self.critic_optimizer = tf.keras.optimizers.Adam(BETA)

    def act(self, state, eps):
        state = tf.expand_dims(state, axis=0)
        values = self.actor_cnn.predict(state, verbose=False)
        dist = tfp.distributions.Categorical(logits=values)
        action = dist.sample()
        return int(action.numpy()[0])

    def step(self, state, action, reward, next_state, done):
        self.learn(state, action, reward, next_state, done)

    @tf.function
    def actor_loss(self, logits, action, td):
        dist = tfp.distributions.Categorical(logits=logits, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor_cnn(state, training=True)
            v = self.critic_cnn(state, training=True)
            vn = self.critic_cnn(next_state, training=True)
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td**2
        grads1 = tape1.gradient(a_loss, self.actor_cnn.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic_cnn.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads1, self.actor_cnn.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(grads2, self.critic_cnn.trainable_variables))
        return a_loss, c_loss

    def save_model(self):
        self.actor_cnn.save_weights("ActorC_model.h5")
        self.critic_cnn.save_weights("ACritic_model.h5")

    def load_model(self):
        self.actor_cnn.load_weights("ActorC_model.h5")
        self.critic_cnn.load_weights("ACritic_model.h5")
