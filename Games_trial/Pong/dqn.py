import collections
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import tqdm

from params import *
from models import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
loss_fn = tf.keras.losses.MeanSquaredError()
model = lenet5()
model.summary()
model.load_weights('lenet_DQN.h5')


def exponential_decay(initial_value, decay_rate, steps_per_decay):
    def exponential_decay_fn(step):
        return initial_value * tf.math.exp(-decay_rate * step / steps_per_decay)
    return exponential_decay_fn


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size=BATCH_SIZE):
        indices = np.random.choice(
            len(self.buffer), batch_size)  # , replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )


def plotting(scores, mean_scores):
    plt.ion()
    plt.clf()
    plt.title('Training Reinforce.')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('TrainingCurve.png')


@tf.function
def train_step(states, actions, rewards, next_states, dones):
    with tf.GradientTape() as tape:
        next_q_values = model(next_states)
        next_q_values = tf.reduce_max(next_q_values, axis=1)
        next_q_values = (1 - dones) * next_q_values
        target_q_values = rewards + next_q_values * GAMMA

        q_values = model(states)
        actions = tf.cast(actions, tf.int32)
        indices = tf.stack([tf.range(BATCH_SIZE), actions], axis=1)
        q_values = tf.gather_nd(q_values, indices)
        loss = loss_fn(target_q_values, q_values)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train(env, episodes=EPISODES, plot=False):
    total_rewards = []
    mean_rewards = []
    replay_buffer = ReplayBuffer(100000)
    epsilon_fn = exponential_decay(
        EPSILON_LEARNING_START, decay_rate=EPSIOLON_DECAY, steps_per_decay=500)

    for episode in tqdm.tqdm(range(EPISODES)):
        state = env.reset()
        state = preprocess(state[0])

        done = False
        total_reward = 0
        while not done:
            if np.random.rand() < epsilon_fn(episode):
                action = np.random.randint(env.action_space.n)
            else:
                state = tf.expand_dims(state, 0)
                action_values = model(state)
                action = np.argmax(action_values)

            next_state, reward, done, _, truncated = env.step(action)
            total_reward += reward
            next_state = preprocess(next_state)

            replay_buffer.add((state, action, reward, next_state, done))
            states, actions, rewards, next_states, dones = replay_buffer.sample()
            train_step(states, actions, rewards, next_states, dones)

            state = next_state

        print('The Total Reward @episodes:', episode, 'is',
              total_reward, 'mean reward', np.mean(total_rewards))

        if (not episode % 100):
            total_rewards.append(total_reward)
            mean = np.mean(total_rewards)
            mean_rewards.append(mean)

            if plot:
                plotting(total_rewards, mean_rewards)

            if mean >= total_reward:
                model.save_weights('lenet_DQN.h5')
