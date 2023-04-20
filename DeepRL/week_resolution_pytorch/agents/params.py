import gymnasium as gym
import argparse
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment_choices = [i for i in gym.envs.registry.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment',
                    help="Environment to train on", type=str, default='BreakoutNoFrameskip-v4', choices=environment_choices)
parser.add_argument('-t', "--train",
                    help="Methods of Training", type=str, default='reinforce', choices=['reinforce', 'reinforce_baseline', 'DQN', 'ActorCritic', 'DoubleDQN', 'DuelingDQN'])
args = parser.parse_args()

MAX_STEPS = 18_000
GAMMA = 0.995
ALPHA = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
NUM_EPISODES = 4_000
EPSILON_DECAY = NUM_EPISODES
MIN_REPLAY_BUFFER_SIZE = 50_000
MEMORY_CAPACITY = 100_000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000
SAVE_FREQUENCY = 10
WINDOW_SIZE = 50


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

input_shape = (4, 84, 84)
n_actions = env.action_space.n
train_mode = args.train
