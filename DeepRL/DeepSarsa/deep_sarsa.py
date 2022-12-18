import gym
import gym_simplegrid
import tensorflow as tf
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

env = gym.make('SimpleGrid-8x8-v0')
my_reward_map = {
    b'E': -1.0,
    b'S': -0.0,
    b'W': -5.0,
    b'G': 5.0,
}
env = gym.make('SimpleGrid-8x8-v0', reward_map=my_reward_map, p_noise=.1)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

EPISODES = 1000


def plot(scores, mean_scores):
    plt.ion()
    plt.clf()
    plt.title('Training Actor Critic.')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('deep_sarsa.png')


class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [0, 1, 2, 3]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 1
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.mse = tf.keras.losses.MeanSquaredError()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    def build_model(self):
        inputs = tf.keras.layers.Input(self.state_size)
        hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(
            self.action_size, activation='linear')(hidden)
        model = tf.keras.Model(inputs, outputs)
        model.summary()
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = tf.cast([state], dtype=tf.float32)
            q_values = self.model(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = tf.cast([state], dtype=tf.float32)
        next_state = tf.cast([next_state], dtype=tf.float32)
        target = self.model(state)[0].numpy()
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model(next_state)[0][next_action])
        target = np.reshape(target, [1, 4])

        with tf.GradientTape() as tape:
            pred = self.model(state)
            loss = self.mse(target, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        # self.model.train_on_batch(state, target)


def main():
    agent = DeepSARSAgent()
    total_rewards = []
    mean_rewards = []
    for game in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info, _ = env.step(action)
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action,
                              done)
            state = next_state
            total_reward += reward
            state = copy.deepcopy(next_state)

            if done:
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(
                    game, total_reward))
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)
        if total_reward > avg_reward:
            agent.model.save_weights('deep_sarsa_trained.h5')
            print('...model save success...')

        mean_rewards.append(avg_reward)
        plot(total_rewards, mean_rewards)


if __name__ == "__main__":
    main()
