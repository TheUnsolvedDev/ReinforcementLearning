import gym
import numpy as np
from collections import defaultdict
import tqdm
import tensorflow as tf
import random

env = gym.make('CartPole-v1')
env = env.unwrapped
n_actions = env.action_space.n


def linear_model(size=4):
    inputs = tf.keras.layers.Input(size)
    hidden = tf.keras.layers.Dense(6,activation = 'relu')(inputs)
    outputs = tf.keras.layers.Dense(n_actions, activation='linear')(hidden)
    model = tf.keras.models.Model(inputs, outputs)
    return model


class Buffer:
    def __init__(self, size=10000):
        self.size = size
        self.storage = []
        self.count = 0

    def push(self, states, values):
        self.storage.append([states, values])
        self.count += 1
        if self.count > self.size:
            self.storage.pop(0)

    def length(self):
        return len(self.storage)

    def sample(self, size=7500):
        batch_size = min(size, self.length())
        return random.sample(self.storage, batch_size)


def monte_carlo(env, episodes=2000, epsilon=1):
    Q = defaultdict(lambda: np.zeros(n_actions))

    state_count = defaultdict(lambda: np.zeros(n_actions))
    epsilon = epsilon
    decay = 1/episodes

    model = linear_model()
    model.compile(loss='mse', optimizer='adam')
    buffer = Buffer()

    for ep in tqdm.tqdm(range(episodes)):
        total_reward = 0

        state_actions = []
        rewards = []

        state = tuple(env.reset())
        done = False
        while not done:
            if np.random.randn() <= epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model(np.array([list(state)])))
            state_actions.append((state, action))
            state, reward, done, null, info = env.step(action)
            state = tuple(state)
            rewards.append(reward)

        for ind in range(len(state_actions)-1, -1, -1):
            S, A = state_actions[ind]
            R = rewards[ind]

            total_reward += R
            if (S, A) not in state_actions[:ind]:
                state_count[S][A] += 1
                Q[S][A] += (total_reward - Q[S][A])/state_count[S][A]

            buffer.push(list(S), Q[S])
        
        if ep % 100 == 50:
            full_data = buffer.sample()
            data = []
            labels = []
            for dat, lab in full_data:
                data.append(dat)
                labels.append(lab)

            data = np.array(data)
            labels = np.array(labels)

            model.fit(data, labels, epochs=50)
            model.save_weights('lin_mod.h5')

        epsilon -= decay

    return Q


def play_games(no_of_games=10, Q=None):
    model = linear_model()
    for ep in range(no_of_games):
        state = env.reset()
        done = False

        total_reward = 0
        while not done:
            action = env.action_space.sample()
            if Q is not None:
                model.load_weights('lin_mod.h5')
                action = np.argmax(model(np.array([state])))
            state, reward, done, null, info = env.step(action)
            total_reward += reward
            env.render()
        print('Cumulative reward:', total_reward, 'Episode no:', ep)


if __name__ == '__main__':
    play_games(10)
    Q = monte_carlo(env)
    print()
    play_games(10,Q = 'lin_mod.h5')

