import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import tensorflow_probability as tfp

env = gym.make("LunarLander-v2", max_episode_steps=500)#, render_mode='human')
gamma = 0.99
alpha = 0.003
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model(in_dim, out_dim):
    inputs = tf.keras.layers.Input(in_dim)
    hidden = tf.keras.layers.Dense(256, activation='relu')(inputs)
    hidden1 = tf.keras.layers.Dense(256, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_dim, activation='softmax')(hidden1)
    return tf.keras.Model(inputs, outputs)


class agent:
    def __init__(self) -> None:
        self.model = model(in_dim, out_dim)
        self.opt = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def act(self, state):
        prob = self.model(np.array([state]))
        categorical = tfp.distributions.Categorical(probs=prob)
        action = categorical.sample()
        return int(action.numpy()[0])

    @tf.function
    def a_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*tf.cast(reward, dtype=tf.float32)
        return tf.reduce_mean(loss)

    def train(self):
        states = tf.convert_to_tensor(self.state_memory, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory)
        actions = tf.convert_to_tensor(self.action_memory)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k]*discount
                discount *= gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            p = self.model(np.array(states), training=True)
            loss = self.a_loss(p, actions, G)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(
            zip(grads, self.model.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


def plot(scores, mean_scores):
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
    plt.savefig('TrainingAndInferenceReinforce.png')


def main():
    agentoo7 = agent()
    total_rewards = []
    mean_rewards = []
    avg_reward = 0
    for game in range(1000):
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            action = agentoo7.act(state)
            next_state, reward, done, info, _ = env.step(action)
            agentoo7.store_transition(state, action, reward)
            state = next_state
            total_reward += reward

            if done:
                agentoo7.train()
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {} avg reward {}".format(
                    game, total_reward, avg_reward))

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)
        if total_reward > avg_reward:
            agentoo7.model.save_weights('reinforce_model.h5')
            print('...model save success...')

        mean_rewards.append(avg_reward)
    plot(total_rewards, mean_rewards)


if __name__ == '__main__':
    main()
