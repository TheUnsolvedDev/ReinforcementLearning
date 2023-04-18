import gymnasium as gym
import argparse
import numpy as np
import cv2


# class ProcessFrame84(gym.ObservationWrapper):
#     def __init__(self, env=None):
#         super(ProcessFrame84, self).__init__(env)
#         self.observation_space = gym.spaces.Box(
#             low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

#     def observation(self, obs):
#         return ProcessFrame84.process(obs)

#     @staticmethod
#     def process(frame):
#         if frame.size == 210 * 160 * 3:
#             img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
#         elif frame.size == 250 * 160 * 3:
#             img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
#         else:
#             assert False, "Unknown resolution."
#         img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
#             0.587 + img[:, :, 2] * 0.114
#         resized_screen = cv2.resize(
#             img, (84, 110), interpolation=cv2.INTER_AREA)
#         x_t = resized_screen[18:102, :]
#         x_t = np.reshape(x_t, [84, 84, 1])
#         return x_t.astype(np.uint8)


environment_choices = [i for i in gym.envs.registry.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment',
                    help="Environment to train on", type=str, default='BreakoutNoFrameskip-v4', choices=environment_choices)
parser.add_argument('-t', "--train",
                    help="Methods of Training", type=str, default='reinforce', choices=['reinforce', 'reinforce_baseline', 'DQN', 'ActorCritic', 'DoubleDQN', 'DuelingDQN'])
args = parser.parse_args()

MAX_STEPS = 18_000
GAMMA = 0.995
ALPHA = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.1
NUM_EPISODES = 4_000
EPSILON_DECAY = 0.995  # (EPSILON_START - EPSILON_END)/(0.5*NUM_EPISODES)
NUM_TRAJECTORIES = 5
MIN_REPLAY_BUFFER_SIZE = 50_000
MEMORY_CAPACITY = 100_000
BATCH_SIZE = 256
UPDATE_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 250
WINDOW_SIZE = 50
SEED = 1234


def make_env(env_id, render_mode=None):
    if render_mode != 'human':
        # , max_episode_steps=MAX_STEPS , render_mode='human')
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode='human',
                       max_episode_steps=MAX_STEPS)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.TransformReward(env, lambda r: 4*r)
    return env


env = make_env(args.environment)
# test_env = gym.make(
#     args.environment, max_episode_steps=MAX_STEPS, render_mode='human')
# test_env = ScaledFloatFrame(ProcessFrame84(test_env))

in_dim = [84, 84, 4]
out_dim = env.action_space.n
train_mode = args.train
