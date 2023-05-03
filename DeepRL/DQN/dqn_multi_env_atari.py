import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium.utils.save_video import save_video

ENV_NAME = "BreakoutDeterministic-v4"
NUM_ENVS = 8
FRAME_STACK = 4
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.025
SIZE = [84, 84]
LEARNING_RATE = 1e-4
Q_UPDATE_FREQ = 4
TARGET_UPDATE_FREQ = 1000
BUFFER_SIZE = 100_000
TOTAL_TIME_STAMPS = 10_000_000
LEARN_START = 10_000

device = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)


def make_train_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array_list")
    env = gym.wrappers.ResizeObservation(env, SIZE)
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, FRAME_STACK)
    env = gym.wrappers.TransformReward(env, lambda r: 2*r)
    return env


vectored_env = gym.vector.AsyncVectorEnv([lambda: make_train_env(
    ENV_NAME) for _ in range(NUM_ENVS)], shared_memory=True)
NUM_ACTIONS = vectored_env.action_space[0].n


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def DQN_Q_model(input_shape=[*SIZE, FRAME_STACK], output_shape=NUM_ACTIONS):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: x/255.0,
                               input_shape=input_shape),
        tf.keras.layers.Conv2D(32, 6, 4, activation='relu'),
        tf.keras.layers.Conv2D(64, 6, 4, activation='relu'),
        tf.keras.layers.Conv2D(64, 4, 1, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_shape)
    ])
    return model


class Replay_buffer:
    def __init__(self) -> None:
        self.memory = []

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) > BUFFER_SIZE:
            self.memory.pop(0)
        self.memory.append([state, action, reward, next_state, done])

    def sample_batch(self, batch_size):
        batch = np.random.choice(self.memory, batch_size)
        return batch


class DQN_agent:
    def __init__(self) -> None:
        self.q_model = DQN_Q_model()
        self.target_q_model = DQN_Q_model()
        self.target_q_model.set_weights(self.q_model.get_weights())
        self.q_model.summary()
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)


def train_agent():
    agent = DQN_agent()


if __name__ == '__main__':
    train_agent()
