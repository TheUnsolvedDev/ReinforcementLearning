import numpy as np
import tensorflow as tf
import gym
import tqdm
import matplotlib.pyplot


class RandomWalk:
    def __init__(self, n_states=1000, n_action=2):
        self.n_states = n_states
        self.n_actions = n_action
        self.reset()

    def reset(self):
        self.start = self.n_states//2
        self.ends = [0, self.n_states-1]
        self.current = self.start
        self.done = False
        return self.current

    def sample_action(self):
        value = np.random.choice([0, 1])
        return value

    def render(self):
        paths = np.zeros(self.n_states)
        paths[self.current] = 1
        print(paths)

    def step(self, action):
        rnd = int(np.random.uniform(0, 100))
        self.movements = int(rnd) if action == 0 else -1*int(rnd)
        if 0 < self.current + self.movements < self.n_states:
            self.current = np.clip(
                self.current + self.movements, 0, self.n_states-1)
        if self.current == 0:
            self.done = True
            reward = -10
        elif self.current == self.ends[1]:
            self.done = True
            reward = 10
        else:
            reward = -1
        state = self.current
        done = self.done
        return state, reward, done, {}, {}


def linear_model(input_shape=(100,)):
    inputs = tf.keras.layers.Input(input_shape)
    outputs = tf.keras.layers.Dense(1)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_state_feature(state, num_states=1000, num_groups=100):
    one_hot_vector = np.zeros(num_groups)
    one_hot_vector[state//(num_states/num_groups)] = 1
    return one_hot_vector

class td_agent:
	def __init__(self):
		pass


if __name__ == '__main__':
    env = RandomWalk()
    state = env.reset()
    done = False
    while not done:
        action = env.sample_action()
        print(state, action)
        state, reward, done, info, _ = env.step(action)
        print(env.movements)
    print(state)
