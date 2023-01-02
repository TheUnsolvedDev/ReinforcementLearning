

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
import sys

from dqn_model import QN, ReplayBuffer

env = gym.make('CartPole-v1')
env = env.unwrapped

# n_states = env.observation_space.n
n_actions = env.action_space.n


plt.ion()


def plot(scores):  # , mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    # plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.001)


def evaluate_training_result(env, agent, show=False, train=True):
    total_reward = 0.0
    episodes_to_play = 30
    for i in range(episodes_to_play):
        state = env.reset()[0]
        done = False
        episode_reward = 0.0
        while not done:
            if not train:
                agent.inference()
            action = agent.policy(state, train=train)

            if show:
                env.render()
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    # env.render()
    return average_reward


def collect_gameplay_experiences(env, agent, buffer):
    state = env.reset()[0]
    done = False
    while not done:
        action = agent.policy(state)
        next_state, reward, done, _, info = env.step(action)
        if done:
            reward = -1.0
        buffer.store_gameplay(state, action, reward, done, next_state)
        state = next_state


def train_model(env, episodes=10000):
    decay = 1/episodes
    gamma = 0.95
    agent = QN(decay=decay)
    buffer = ReplayBuffer()
    average_rewards = []
    # total_epicodes = []
    for eps in range(episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experiences_batch = buffer.sample_batch()
        loss = agent.train(gameplay_experiences_batch)
        agent.epsilon = max(0.0001, agent.epsilon - agent.decay)
        average_reward = evaluate_training_result(env, agent)
        average_rewards.append(average_reward)

        plot(average_rewards)
        print('\rEpisode {0}/{1} and so far the performance is {2} and '
              'loss is {3} with epsilon {4}'.format(eps, episodes,
                                                    average_reward, loss[0], agent.epsilon), end='')
        sys.stdout.flush()
        if eps % int(episodes/10) == 0:
            average_reward = evaluate_training_result(env, agent, show=True)
            agent.target_Q.set_weights(agent.Q.get_weights())
            agent.target_Q.save('Q.h5')
            print()


if __name__ == '__main__':
    train_model(env)
    agent = QN(decay=0.1)
    evaluate_training_result(env, agent, show=True, train=False)
    env.close()
