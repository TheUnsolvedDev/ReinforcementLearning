import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import tensorflow_probability as tfp

env = gym.make('CartPole-v1')
env.max_episode_steps = 1000
test_env = gym.make('CartPole-v1', render_mode='human')

gamma = 0.99
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model(in_dim, out_dim):
    inputs = tf.keras.layers.Input(in_dim)
    hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_dim, activation='linear')(hidden1)
    return tf.keras.Model(inputs, outputs)


class agent:
    def __init__(self) -> None:
        self.model = model(in_dim, out_dim)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    def act(self, state):
        prob = self.model(np.array([state]))
        categorical = tfp.distributions.Categorical(logits=prob)
        action = categorical.sample()
        return int(action.numpy()[0])

    def a_log_prob(self, prob, action):
        dist = tfp.distributions.Categorical(logits=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        return log_prob

    def train(self, states, rewards, actions):
        raise NotImplementedError
        discnt_rewards = []
        for t in range(len(rewards)):  # reward-to-go
            sum_reward = 0
            discount = 1
            for k in range(t, len(rewards)):
                sum_reward += rewards[k]*discount
                discount *= gamma
            discnt_rewards.append(sum_reward)

        for state, reward, action in zip(states, discnt_rewards, actions):
            p_log_prob = 0
            q_log_prob = 0

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([state]), training=True)
                log_prob = self.a_log_prob(p, action)
                loss = -log_prob*reward
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(
                zip(grads, self.model.trainable_variables))


def plot(scores, mean_scores):
    plt.ion()
    plt.clf()
    plt.title('Training Reinforce.')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('TrainingAndInferenceReinforce.png')


def main():
    agentoo7 = agent()
    total_rewards = []
    mean_rewards = []
    for game in range(1000):
        state = env.reset()[0]
        total_reward = 0
        rewards = []
        states = []
        actions = []
        done = False
        while not done:
            action = agentoo7.act(state)
            next_state, reward, done, info, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward += reward

            if done:
                agentoo7.train(states, rewards, actions)
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(
                    game, total_reward))

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)
        if total_reward > avg_reward:
            agentoo7.model.save_weights('reinforce_model.h5')
            print('...model save success...')

        mean_rewards.append(avg_reward)
        plot(total_rewards, mean_rewards)


if __name__ == '__main__':
    main()
