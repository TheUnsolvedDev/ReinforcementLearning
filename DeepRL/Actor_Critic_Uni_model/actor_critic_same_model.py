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


def actor_critic(in_dim=in_dim, out_dim=out_dim):
    inputs = tf.keras.layers.Input(in_dim)
    hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs1 = tf.keras.layers.Dense(out_dim, activation='softmax')(hidden1)
    outputs2 = tf.keras.layers.Dense(1, activation='linear')(hidden1)
    return tf.keras.Model(inputs, [outputs2, outputs1])


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
    plt.savefig('TrainingAndInferenceActorCriticV2.png')


class agent():
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.ac = actor_critic()

    def act(self, state):
        v, prob = self.ac(np.array([state]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])

        with tf.GradientTape() as tape:
            v, a = self.ac(state, training=True)
            vn, an = self.ac(next_state, training=True)
            td = reward + self.gamma*vn*(1-int(done)) - v
            a_loss = self.actor_loss(a, action, td)
            c_loss = td**2
            total_loss = a_loss + c_loss
        grads = tape.gradient(total_loss, self.ac.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.ac.trainable_variables))
        return total_loss


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
