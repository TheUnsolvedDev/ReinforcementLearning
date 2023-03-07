import numpy as np
import tensorflow as tf
import gymnasium as gym
import tqdm

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


NUM_STATES = 1000
NUM_ACTIONS = 2
NUM_GROUPS = 100


class RandomWalk:
    def __init__(self, n_states=NUM_STATES, n_action=NUM_ACTIONS):
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


def get_state_feature(state, num_states=NUM_STATES, num_groups=NUM_GROUPS):
    one_hot_vector = np.zeros(num_groups)
    loc = int(state//(num_states/num_groups))
    one_hot_vector[loc] = 1
    return one_hot_vector


def neural_network(input_shape=(NUM_GROUPS,)):
    inputs = tf.keras.layers.Input(input_shape)
    hidden = tf.keras.layers.Dense(
        16, activation='relu')(inputs)
    output = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs, output)
    return model

def semi_gradient_mc(env, iterations=200):
    alpha = 0.1
    gamma = 1
    nn = neural_network()
    optimizer = tf.keras.optimizers.Adam()
    for iter in tqdm.tqdm(range(iterations)):
        done = env.done
        state = env.reset()
        S = get_state_feature(state)
        states = []
        rewards = []
        while not done:
            action = np.random.choice([0, 1])
            states.append(state)
            state, reward, done, info, _ = env.step(action)
            S = get_state_feature(state)
            rewards.append(reward)

        total_reward = 0
        for ind in range(len(states)):
            total_reward += rewards[ind]
            state = states[ind]
            S = get_state_feature(state)
            with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
                current_value= nn(tf.expand_dims(S, axis = 0))
                delta = total_reward - current_value
            grads = tape.gradient(delta, nn.trainable_variables)
            optimizer.apply_gradients(zip(grads, nn.trainable_variables))

    nn.save_weights('nn.h5')


if __name__ == '__main__':
    env = RandomWalk()
    semi_gradient_mc(env)
