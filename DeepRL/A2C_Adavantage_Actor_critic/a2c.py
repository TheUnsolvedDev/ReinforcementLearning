import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

# env = gym.make("LunarLander-v2")
env = gym.make('CartPole-v1')
env.max_episode_steps = 1000
low = env.observation_space.low
high = env.observation_space.high
gamma = 0.99
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def actor(in_dim=in_dim, out_dim=out_dim):
    inputs = tf.keras.layers.Input(in_dim)
    hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_dim, activation='softmax')(hidden1)
    return tf.keras.Model(inputs, outputs)


def critic(in_dim=in_dim):
    inputs = tf.keras.layers.Input(in_dim)
    hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(1, activation='linear')(hidden1)
    return tf.keras.Model(inputs, outputs)


class agent():
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):
        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        p_loss = []
        e_loss = []
        td = td.numpy()
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        return loss


def main():
    agentoo7 = agent()
    total_rewards = []
    mean_rewards = []
    for game in range(1000):
        state = env.reset()[0]
        total_reward = 0
        all_loss = []
        done = False
        while not done:
            action = agentoo7.act(state)
            next_state, reward, done, info, _ = env.step(action)
            loss = agentoo7.learn(
                state, action, reward, next_state, done)
            all_loss.append(loss)
            state = next_state
            total_reward += reward

            if done:
                #print("total step for this episord are {}".format(t))
                print("total reward after {} steps is {}".format(
                    game, total_reward))
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)
        if total_reward > avg_reward:
            agentoo7.ac.save_weights('actor_critic_ac_model.h5')
            print('...model save success...')

        mean_rewards.append(avg_reward)
        plot(total_rewards, mean_rewards)


if __name__ == '__main__':
    main()
