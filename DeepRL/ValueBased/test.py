from silence_tensorflow import silence_tensorflow
import tensorflow as tf
import gymnasium as gym
import numpy as np
import argparse
import random

tf.keras.backend.set_floatx('float32')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(1234)
silence_tensorflow()

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='CartPole-v1')
parser.add_argument('--max_steps', type=int, default=1_000)
args = parser.parse_args()


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = 0.01

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


def main():
    env = gym.make(args.env_name, render_mode='human',
                   max_episode_steps=args.max_steps)
    agent = ActionStateModel(
        env.observation_space.shape[0], env.action_space.n)
    agent.model.load_weights('weights/DQN_weights_'+args.env_name+'.h5')

    for ep in range(10):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            state, reward, done, truncated, _ = env.step(action)
        print('Episode {}: {}'.format(ep, reward))


if __name__ == '__main__':
    main()
