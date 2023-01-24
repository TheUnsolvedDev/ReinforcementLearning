
from stack_frame import *
from agents import *
from params import *
import gymnasium as gym
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
import argparse
import gc
from silence_tensorflow import silence_tensorflow
from collections import deque

silence_tensorflow()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def epsilon_by_epsiode(frame_idx): return EPS_END + \
    (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)


def train(agent, env, n_episodes=50000, stacking_frames=False):
    start_epoch = 0
    scores = []
    scores_window = deque(maxlen=20)

    for i_episode in tqdm.tqdm(range(start_epoch + 1, n_episodes+1)):
        gc.collect()
        if stacking_frames:
            state = stack_frames(None, preprocess_frame(env.reset()[0]), True)
        else:
            state = env.reset()[0]
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, trauncated, info = env.step(action)
            score += reward

            if stacking_frames:
                next_state = preprocess_frame(next_state)
                next_state = stack_frames(state, next_state, False)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done or trauncated:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")

        if i_episode % 50 == 0:
            agent.save_model()
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
            

    return scores


if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode='rgb_array')  # 'human')
    # env.metadata['render_fps'] = 144

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='Choose training agent.',
                        default='DQN_agent', choices=['DQN_agent', 'DoubleDQN_agent', 'DuelingDQN_agent', 'Reinforce_agent', 'Reinforce_agent_baseline', 'ActorCritic_agent'])
    parser.add_argument('--batch_size', help='Number of training batches', type=int,
                        default=128, choices=[16, 32, 64, 128, 256, 512])
    parser.add_argument('--stack_frames', help='To stack Frames as Deepmind',
                        default='False', choices=['False', 'True'])
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
    train(agent, env, stacking_frames=False)
    env.close()
