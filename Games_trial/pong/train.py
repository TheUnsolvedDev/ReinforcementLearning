
from stack_frame import *
from agents import *
from params import *
import gym
import random
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import argparse
from silence_tensorflow import silence_tensorflow
from collections import deque

silence_tensorflow()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

env = gym.make('PongNoFrameskip-v4', render_mode='rgb_array')  # 'human')
env.metadata['render_fps'] = 144
env.reset()


def random_play():
    score = 0
    state = env.reset()[0]
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _, truncated = env.step(action)
        score += reward
        if done:
            env.close()
            print("Your Score at end of game is: ", score)
            break


def epsilon_by_epsiode(frame_idx): return EPS_END + \
    (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)


def train(agent, n_episodes=50000):
    start_epoch = 0
    scores = []
    scores_window = deque(maxlen=20)

    for i_episode in tqdm.tqdm(range(start_epoch + 1, n_episodes+1)):
        state = stack_frames(None, preprocess_frame(env.reset()[0]), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, info, trauncated = env.step(action)
            score += reward
            next_state = preprocess_frame(next_state)
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")

        if i_episode % 50 == 0:
            agent.save_model()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # plt.plot(np.arange(len(scores)), scores)
            # plt.ylabel('Score')
            # plt.xlabel('Episode #')
            # plt.show()

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='Choose training agent.',
                        default='DQN_agent', choices=['DQN_agent', 'DoubleDQN_agent', 'DuelingDQN_agent', 'Reinforce_agent', 'Reinforce_agent_baseline', 'ActorCritic_agent'])
    parser.add_argument('--batch_size', help='Number of training batches', type=int,
                        default=128, choices=[16, 32, 64, 128, 256, 512])
    args = parser.parse_args()

    print()
    print('***** Training Agent:', args.agent, '*****')
    print()

    if args.agent == 'DQN_agent':
        agent = DeepQNetwork_agent(sample_size=args.batch_size)
    if args.agent == 'DoubleDQN_agent':
        agent = DoubleDeepQNetwork_agent(sample_size=args.batch_size)
    if args.agent == 'DuelingDQN_agent':
        agent = DoubleDeepQNetwork_agent(sample_size=args.batch_size)
    if args.agent == 'Reinforce_agent':
        agent = Reinforce_agent(sample_size=args.batch_size)
    if args.agent == 'Reinforce_agent_baseline':
        agent = Reinforce_agent(
            sample_size=args.batch_size, use_baseline=True)
    if args.agent == 'ActorCritic_agent':
        agent = ActorCritic_agent(sample_size=args.batch_size)

    try:
        agent.load_model()
    except FileNotFoundError:
        pass
    train(agent)
