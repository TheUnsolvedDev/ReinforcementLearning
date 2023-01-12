import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

MAX_EPISODES = 1000
MAX_STEPS = 50000
# env = gym.make("LunarLander-v2", max_episode_steps=500,
#                enable_wind=True, render_mode='human')
env = gym.make('CartPole-v1', max_episode_steps=500, render_mode='human')

gamma = 0.99
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n


def model(in_dim=in_dim, out_dim=out_dim):
    inputs = tf.keras.layers.Input(in_dim)
    hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_dim, activation='linear')(hidden1)
    return tf.keras.Model(inputs, outputs)


def action_value(model, states):
    return model(states)


class experience():
    def __init__(self, buffer_size, state_dim):
        self.buffer_size = buffer_size
        self.pointer = 0
        self.state_mem = np.zeros(
            (self.buffer_size, *state_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.buffer_size, dtype=np.int32)
        self.next_state_mem = np.zeros(
            (self.buffer_size, *state_dim), dtype=np.float32)
        self.reward_mem = np.zeros(self.buffer_size, dtype=np.int32)
        self.done_mem = np.zeros(self.buffer_size)

    def add_exp(self, state, action, reward, next_state, done):
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = done
        self.pointer += 1

    def sample_exp(self, batch_size):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        state = self.state_mem[batch]
        action = self.action_mem[batch]
        reward = self.reward_mem[batch]
        next_state = self.next_state_mem[batch]
        done = self.done_mem[batch]
        return state, action, reward, next_state, done


class agent():
    def __init__(self):
        self.q_net = model()
        self.target_net = model()
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.q_net.compile(optimizer=opt, loss='mse')
        self.target_net.compile(optimizer=opt, loss='mse')
        self.epsilon = 1.0
        self.epsilon_decay = 1/(0.6*MAX_EPISODES)  # 1e-4*5
        self.min_epsilon = 0.01
        self.memory = experience(
            buffer_size=1000000, state_dim=env.observation_space.shape)
        self.batch_size = 64
        self.gamma = 0.99
        self.replace = 20
        self.trainstep = 0
        self.action_space = [i for i in range(out_dim)]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([state])
            action = action_value(self.q_net, state)
            action = np.argmax(action)
        return action

    def train(self):
        if self.memory.pointer < self.batch_size:
            return
        if self.trainstep % self.replace == 0:
            self.update_target()

        states, actions, rewards, next_states, dones = self.memory.sample_exp(
            self.batch_size)
        target = action_value(self.q_net, states)
        next_state_val = action_value(self.target_net, next_states)
        q_next = tf.math.reduce_max(
            next_state_val, axis=1, keepdims=True).numpy()
        q_target = np.copy(target)
        for i, d in enumerate(dones):
            if d:
                q_target[i, actions[i]] = rewards[i]
            else:
                q_target[i, actions[i]] = rewards[i] + self.gamma * q_next[i]
        self.q_net.train_on_batch(states, q_target)
        # self.update_epsilon()
        self.trainstep += 1

    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - \
            self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def save_model(self):
        self.q_net.save_weights("model.h5")
        self.target_net.save_weights("target_model.h5")

    def load_model(self):
        self.q_net.load_weights("target_model.h5")
        self.target_net.load_weights("target_model.h5")


def plot(scores, mean_scores):
    plt.ion()
    plt.clf()
    plt.title('Training DQN.')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('vanillaDQN.png')


def main():
    agentoo7 = agent()
    # agentoo7.load_model()
    # agentoo7.epsilon = 0
    total_rewards = []
    mean_rewards = []
    avg_reward = 0
    for game in range(MAX_EPISODES):
        state = env.reset()[0]
        done = False
        total_reward = 0
        t = 0
        while not done:
            if t > MAX_STEPS:
                break
            action = agentoo7.act(state)
            next_state, reward, done, info, _ = env.step(int(action))
            agentoo7.update_mem(state, action, reward, next_state, done)
            agentoo7.train()
            state = next_state
            total_reward += reward
            t += 1
            if done:
                print("total reward after {} steps is {} avg reward {}".format(
                    game, total_reward, avg_reward))
        agentoo7.update_epsilon()
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards)
        if total_reward > avg_reward:
            agentoo7.save_model()
            print('...model save success...')

        mean_rewards.append(avg_reward)
    plot(total_rewards, mean_rewards)


if __name__ == '__main__':
    main()
