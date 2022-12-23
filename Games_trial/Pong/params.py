import gym
import tensorflow as tf

EPSILON_LEARNING_START = 1
EPSILON_LEARNING_END = 0.01
EPSILON = 0.1
EPSIOLON_DECAY = 0.01
GAMMA = 0.99
ALPHA = 0.01
IMG_SIZE = (64, 64)
EPISODES = 100000

USEFUL_REGION_TOP = 34
USEFUL_REGION_BOTTOM = 192
BATCH_SIZE = 32

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def preprocess(state):
    state = state[USEFUL_REGION_TOP: USEFUL_REGION_BOTTOM, :, :]
    state = tf.image.resize(state, IMG_SIZE)
    return tf.expand_dims(tf.reduce_mean(state, axis=-1), axis=-1)


env = gym.make('PongNoFrameskip-v4')
# test_env = gym.make('PongNoFrameskip-v4', render_mode='human')
