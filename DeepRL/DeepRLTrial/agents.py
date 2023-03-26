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

<<<<<<< Updated upstream
<<<<<<< Updated upstream
        self.actor_cnn = dnn(INPUT_SHAPE, ACTION_SIZE)
        self.critic_cnn = dnn(INPUT_SHAPE, 1)
=======
        self.actor_model = dnn(INPUT_SHAPE, ACTION_SIZE)
        self.critic_model = dnn(INPUT_SHAPE, 1)
>>>>>>> Stashed changes
=======
        self.actor_model = dnn(INPUT_SHAPE, ACTION_SIZE)
        self.critic_model = dnn(INPUT_SHAPE, 1)
>>>>>>> Stashed changes

        self.actor_optimizer = tf.keras.optimizers.Adam(ALPHA)
        self.critic_optimizer = tf.keras.optimizers.Adam(BETA)

    def act(self, state, eps):
        state = tf.expand_dims(state, axis=0)
        values = self.actor_model.predict(state, verbose=False)
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
            p = self.actor_model(state, training=True)
            v = self.critic_model(state, training=True)
            vn = self.critic_model(next_state, training=True)
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td**2
        grads1 = tape1.gradient(a_loss, self.actor_model.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads1, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(
            zip(grads2, self.critic_model.trainable_variables))
        return a_loss, c_loss

    def save_model(self):
        self.actor_model.save_weights("ActorC_model.h5")
        self.critic_model.save_weights("ACritic_model.h5")

    def load_model(self):
        self.actor_model.load_weights("ActorC_model.h5")
        self.critic_model.load_weights("ACritic_model.h5")


class DeepQNetwork_agent:
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.sample_size = sample_size

        self.model = dnn(INPUT_SHAPE, ACTION_SIZE)
        self.target_model = dnn(INPUT_SHAPE, ACTION_SIZE)
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
            return np.argmax(self.model.predict(state, verbose=False))

    @tf.function
    def get_loss(self, q_expected, rewards, q_targets, dones):
        rewards = tf.cast(rewards, tf.float32)
        q_targets = tf.add(rewards, (self.gamma)*q_targets*(1.-dones))
        losses = self.loss(q_targets, q_expected)
        return losses

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape1:
            q_values = self.model(states)
            next_q_values = tf.stop_gradient(self.target_model(next_states))
            q_expected = tf.math.reduce_sum(
                q_values * tf.one_hot(actions, ACTION_SIZE), axis=1)
            q_targets = tf.math.reduce_max(next_q_values, axis=1)
            losses = self.get_loss(
                q_expected, rewards, q_targets, dones)

        grads = tape1.gradient(losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        self.soft_update()

    def soft_update(self):
        for target, policy in zip(self.target_model.layers, self.model.layers):
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
        self.model.save_weights("DQN_model.h5")
        self.target_model.save_weights("DQN_target_model.h5")

    def load_model(self):
        self.model.load_weights("DQN_target_model.h5")
        self.target_model.load_weights("DQN_target_model.h5")


class DoubleDeepQNetwork_agent(DeepQNetwork_agent):
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        super().__init__(epsilon, gamma, memory_capacity, sample_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape1:
            q_values = self.model(states)
            next_q_values = tf.stop_gradient(self.target_model(next_states))
            q_next_values = self.model(next_states)

            q_expected = tf.reduce_sum(
                q_values * tf.one_hot(actions, ACTION_SIZE), axis=1)
            next_actions = tf.argmax(q_next_values, axis=1)
            q_targets = tf.reduce_sum(
                next_q_values*tf.one_hot(actions, ACTION_SIZE), axis=1)

            rewards = tf.cast(rewards, tf.float32)
            q_targets = tf.add(rewards, (self.gamma)*q_targets*(1.-dones))
            losses = self.loss(q_targets, q_expected)

        grads = tape1.gradient(losses, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    def save_model(self):
        self.model.save_weights("DDQN_model.h5")
        self.target_model.save_weights("DDQN_target_model.h5")

    def load_model(self):
        self.model.load_weights("DDQN_target_model.h5")
        self.target_model.load_weights("DDQN_target_model.h5")


class DuelingDQN_agent(DeepQNetwork_agent):
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE):
        super().__init__(epsilon, gamma, memory_capacity, sample_size)


class Reinforce_agent:
    def __init__(self, epsilon=EPS_START, gamma=GAMMA, memory_capacity=BUFFER_SIZE, sample_size=BATCH_SIZE, use_baseline=False):
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.sample_size = sample_size
        self.use_baseline = use_baseline

        self.model = cnn(INPUT_SHAPE, ACTION_SIZE)
        self.optimizer = tf.keras.optimizers.Adam(ALPHA)
        self.memory = ReplayBuffer(BUFFER_SIZE)

        if self.use_baseline:
            self.baseline_model = cnn(INPUT_SHAPE, 1)
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
        values = self.model.predict(state, verbose=False)
        dist = tfp.distributions.Categorical(logits=values)
        action = dist.sample()
        return int(action.numpy()[0])

    def get_advantage(self, returns, observations):
        values = self.baseline_model(observations)
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
                predictions = self.baseline_model(states)
                loss = self.advantage_loss(returns, predictions)
            grads = tape.gradient(loss, self.baseline_model.trainable_weights)
            self.baseline_optimizer.apply_gradients(
                zip(grads, self.baseline_model.trainable_weights))
            returns = advantages

        with tf.GradientTape() as tape:
            values = self.model(states)
            loss = self.get_loss(values, actions, returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        self.reset_memory()

    def save_model(self):
        if self.use_baseline:
            self.model.save_weights("PG_baseline_model.h5")
        else:
            self.model.save_weights("PG_model.h5")

    def load_model(self):
        if self.use_baseline:
            self.model.load_weights("PG_baseline_model.h5")
        else:
            self.model.load_weights("PG_model.h5")
