import gymnasium as gym
import argparse

environment_choices = [i for i in gym.envs.registry.keys()]

parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment',
                    help="Environment to train on", type=str, default='CartPole-v1', choices=environment_choices)
parser.add_argument('-t', "--train",
                    help="Methods of Training", type=str, default='reinforce', choices=['reinforce', 'reinforce_baseline', 'DQN', 'ActorCritic', 'DoubleDQN', 'DuelingDQN'])
args = parser.parse_args()

MAX_STEPS = 1000
env = gym.make(args.environment, max_episode_steps=MAX_STEPS)
test_env = gym.make(
    args.environment, max_episode_steps=MAX_STEPS, render_mode='human')
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n
train_mode = args.train

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
