import gymnasium as gym
import argparse
import numpy as np
import cv2


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = ProcessFrame84(env)
    return ScaledFloatFrame(env)


environment_choices = [i for i in gym.envs.registry.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment',
                    help="Environment to train on", type=str, default='CartPole-v1', choices=environment_choices)
parser.add_argument('-t', "--train",
                    help="Methods of Training", type=str, default='reinforce', choices=['reinforce', 'reinforce_baseline', 'DQN', 'ActorCritic', 'DoubleDQN', 'DuelingDQN'])
args = parser.parse_args()

MAX_STEPS = 1000
GAMMA = 0.995
ALPHA = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.01
NUM_EPISODES = 2000
EPSILON_DECAY = (EPSILON_START - EPSILON_END)/(0.5*NUM_EPISODES)
NUM_TRAJECTORIES = 10
MEMORY_CAPACITY = 100000
BATCH_SIZE = 256
TARGET_UPDATE_FREQUENCY = 10
WINDOW_SIZE = 50
SEED = 1234

env = gym.make(args.environment, max_episode_steps=MAX_STEPS)
env = ScaledFloatFrame(ProcessFrame84(env))

# test_env = gym.make(
#     args.environment, max_episode_steps=MAX_STEPS, render_mode='human')
# test_env = ScaledFloatFrame(ProcessFrame84(test_env))

in_dim = [84, 84, 4]
out_dim = 6
train_mode = args.train
