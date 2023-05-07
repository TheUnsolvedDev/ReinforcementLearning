import wandb
from silence_tensorflow import silence_tensorflow
import tensorflow as tf
import gymnasium as gym
import argparse
import numpy as np
from collections import deque
import random
import tqdm

tf.keras.backend.set_floatx('float32')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(1234)
silence_tensorflow()

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.00025)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.025)
parser.add_argument('--memory_size', type=int, default=10_000)
parser.add_argument('--max_steps', type=int, default=1_000)
parser.add_argument('--max_episodes', type=int, default=10_000)
parser.add_argument('--env_name', type=str, default='CartPole-v1')

args = parser.parse_args()
config = {
    'gamma': args.gamma,
    'lr': args.lr,
    'batch_size': args.batch_size,
    'eps': args.eps,
    'eps_decay': args.eps_decay,
    'eps_min': args.eps_min,
    'memory_size': args.memory_size,
    'max_steps': args.max_steps,
    'max_episodes': args.max_episodes,
    'env_name': args.env_name
}


wandb.init(name='DQN_'+config['env_name'],
           project="clean_RL", config=config)


class ReplayBuffer:
    def __init__(self, capacity=args.memory_size):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states, dtype=np.float32).reshape(
            args.batch_size, -1)
        next_states = np.array(next_states, dtype=np.float32).reshape(
            args.batch_size, -1)
        done = np.array(done, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps

        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input((self.state_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        return model

    @tf.function
    def predict(self, state):
        state = tf.expand_dims(state, axis=0)
        return self.model(state)

    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = tf.keras.optimizers.Adam(args.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, done):
        with tf.GradientTape() as tape:
            targets = self.target_model.model(states)
            next_q_values = tf.reduce_max(
                self.target_model.model(next_states, training=False), axis=1)
            actions = tf.cast(actions, tf.int32)
            indices = tf.stack([tf.range(args.batch_size), actions], axis=1)
            updates = rewards + (1.0-done) * next_q_values * args.gamma
            targets = tf.tensor_scatter_nd_update(targets, indices, updates)
            pred = self.model.model(states)

            loss = tf.reduce_mean(self.loss_fn(targets, pred))
        grads = tape.gradient(loss, self.model.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.model.trainable_variables))
        return loss

    def replay(self):
        loss = 0
        for _ in range(10):
            loss_value = self.train_step(*self.buffer.sample())
        loss += loss_value
        return loss

    def train(self, max_episodes=args.max_episodes):
        loss = 0
        rewards_collection = []
        for ep in tqdm.tqdm(range(max_episodes)):
            done, total_reward = False, 0
            state = self.env.reset()[0]
            for step in range(args.max_steps):
                action = self.model.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            rewards_collection.append(total_reward)
            if self.buffer.size() >= args.batch_size:
                loss = self.replay()
            self.target_update()
            self.target_model.model.save_weights(
                'weights/DQN_weights_'+config['env_name']+'.h5')
            wandb.log({'Loss': loss})
            wandb.log({'Reward': total_reward})
            wandb.log({'Average Reward': np.mean(rewards_collection[-100:])})
            wandb.log({'Epsilon': self.model.epsilon})
            wandb.log({'Buffer Size': self.buffer.size()})


def main():
    env = gym.make(config['env_name'], max_episode_steps=args.max_steps)
    agent = Agent(env)
    agent.train()


if __name__ == "__main__":
    main()

wandb.finish()
