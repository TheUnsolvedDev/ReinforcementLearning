import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from params import *
from model import *
from replay_buffer import *


class DQN_agent:
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

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape1:
            q_values = self.cnn(states)
            next_q_values = tf.stop_gradient(self.target_cnn(next_states))

            q_expected = tf.reduce_sum(
                q_values * tf.one_hot(actions, 6), axis=1)
            q_targets = tf.reduce_max(next_q_values, axis=1)
            rewards = tf.cast(rewards, tf.float32)
            q_targets = tf.add(rewards, (self.gamma)*q_targets*(1.-dones))
            losses = self.loss(q_targets, q_expected)

        grads = tape1.gradient(losses, self.cnn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.cnn.trainable_variables))

        self.soft_update()

    def soft_update(self):
        for target, policy in zip(self.target_cnn.layers[1:], self.cnn.layers[1:]):
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


class DDQN_agent(DQN_agent):
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

    def learn(self):
        returns = tf.convert_to_tensor(
            self.compute_returns(0), dtype=tf.float32)
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)

        if self.use_baseline:
            advantages = self.get_advantage(returns, states)
            with tf.GradientTape() as tape:
                predictions = self.baseline_cnn(ostates)
                loss = tf.keras.losses.mean_squared_error(
                    y_true=returns, y_pred=predictions)
            grads = tape.gradient(loss, self.baseline_cnn.trainable_weights)
            self.baseline_optimizer.apply_gradients(
                zip(grads, self.baseline_cnn.trainable_weights))
            returns = advantages

        with tf.GradientTape() as tape:
            values = self.cnn(states)
            dist = tfp.distributions.Categorical(logits=values)
            log_prob = dist.log_prob(actions)
            loss = -tf.reduce_mean(log_prob * returns)

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


class ActorCritic:
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.sample_size = sample_size

        self.actor_cnn = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.critic_cnn = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.actor_optimizer = tf.keras.optimizers.Adam(ALPHA)
        self.critic_optimizer = tf.keras.optimizers.Adam(BETA)

        self.states = []
        self.rewards = []
        self.actions = []
        self.masks = []

    def reset_memory(self):
        del self.states[:]
        del self.rewards[:]
        del self.actions[:]
        del self.masks[:]

    def act(self, state, eps):
        state = tf.expand_dims(state, axis=0)
        values = self.cnn.predict(state, verbose=False)
        dist = tfp.distributions.Categorical(logits=values)
        action = dist.sample()
        return int(action.numpy()[0])

    def step(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(1 - done)
        if done:
            # print('Starting to learn...')
            self.learn()

    def compute_returns(self, next_value):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns

    def learn(self):
        self.reset_memory()

    def save_model(self):
        self.cnn.save_weights("AC_model.h5")

    def load_model(self):
        self.cnn.load_weights("AC_model.h5")


if __name__ == '__main__':
    pass
