import gymnasium as gym
import argparse

environment_choices = [i for i in gym.envs.registry.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment',
                    help="Environment to train on", type=str, default='CartPole-v1', choices=environment_choices)
parser.add_argument('-t', "--train",
                    help="Methods of Training", type=str, default='reinforce', choices=['reinforce', 'reinforce_baseline', 'DQN', 'ActorCritic', 'DoubleDQN','DuelingDQN'])
args = parser.parse_args()

MAX_STEPS = 1000
env = gym.make(args.environment, max_episode_steps=MAX_STEPS)
test_env = gym.make(
    args.environment, max_episode_steps=MAX_STEPS, render_mode='human')
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n
train_mode = args.train

GAMMA = 0.995
ALPHA = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.05
NUM_EPISODES = 10000
EPSILON_DECAY = 0.999
NUM_TRAJECTORIES = 10
MEMORY_CAPACITY = 5000
BATCH_SIZE = 256
TARGET_UPDATE_FREQUENCY = 5
