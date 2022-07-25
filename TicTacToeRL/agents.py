
from operator import ne
import matplotlib.pyplot as plt
import numpy as np
import logging
from collections import defaultdict
import tqdm
import json
import pandas as pd

from params import *
from env import TicTacToe, agent_by_mark, check_game_status, \
    next_state_show, code_to_mark, next_mark

plt.ion()


class BaseAgent:
    def __init__(self, mark) -> None:
        self.mark = mark

    def act(self, state, actions):
        for action in actions:
            n_state = next_state_show(state, action)
            stats = check_game_status(n_state[0])
            if stats > 0:
                if code_to_mark(stats) == self.mark:
                    return action
        return np.random.choice(actions)


class HumanAgent:
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action


def plot(scores1, scores2):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores1)
    plt.plot(scores2)
    plt.text(len(scores1)-1, scores1[-1], str(scores1[-1]))
    plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    plt.show(block=False)
    plt.pause(0.001)


V = {}
visit = defaultdict(int)


def set_state_value(state, value):
    visit[state] += 1
    V[state] = value


class TDAgent:
    def __init__(self, mark, epsilon, alpha, train=True, V=None):
        self.mark = mark
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_actions = NUM_LOC
        self.V = V

    def act(self, state, ava_actions, full_greedy=False):
        if self.V:
            if self.mark == 'O':
                ava_values = [self.V[next_state_show(
                    state, action)] for action in ava_actions]
                best = np.max(ava_values)
            else:
                ava_values = [self.V[next_state_show(
                    state, action)] for action in ava_actions]
                best = np.min(ava_values)
            indices = [i for i, v in enumerate(ava_values) if v == best]
            aidx = np.random.choice(indices)
            action = ava_actions[aidx]
            return action
        return self.greedy(state, ava_actions, full_greedy)

    def greedy(self, state, ava_actions, full_greedy=False):
        logging.debug("egreedy_policy for '{}'".format(self.mark))

        e = np.random.uniform(low=0, high=self.epsilon)
        if e < self.epsilon:
            logging.debug("Explore with eps {}".format(self.epsilon))
            action = self.random_action(ava_actions)
        else:
            logging.debug("Exploit with eps {}".format(self.epsilon))
            action = self.greedy_action(state, ava_actions)
        if full_greedy:  # it is for the sample game
            action = self.greedy_action(state, ava_actions)
        return action

    def random_action(self, ava_actions):
        return np.random.choice(ava_actions)

    def greedy_action(self, state, ava_actions):
        assert len(ava_actions) > 0
        ava_values = []
        for action in ava_actions:
            next_state = next_state_show(state, action)
            next_val = self.retake_val(next_state)
            ava_values.append(next_val)

        if self.mark == 'O':
            best = np.max(ava_values)
        if self.mark == 'X':
            best = np.min(ava_values)
        indices = [i for i, v in enumerate(ava_values) if v == best]
        aidx = np.random.choice(indices)
        action = ava_actions[aidx]
        return action

    def retake_val(self, state):
        if state not in V:
            stats = check_game_status(state[0])
            val = DEFAULT_VALUE

            if stats > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return V[state]

    def learn(self, state, next_state, reward):
        val = self.retake_val(state)
        next_val = self.retake_val(next_state)

        diff = next_val - val
        new_val = val + self.alpha * diff
        set_state_value(state, new_val)


def save_model(save_file, V, max_episode, epsilon, alpha):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="td", max_episode=max_episode, epsilon=epsilon,
                    alpha=alpha)
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for state, value in V.items():
            f.write('{}\t{}\n'.format(state, value))


def load_model(filename):
    V = {}
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            V[state] = val
    return V


def sample_game(td_agent):
    start_mark = 'O'
    env = TicTacToe()
    results = []
    if td_agent.mark == 'O':
        agents = [td_agent, BaseAgent('X')]
    else:
        agents = [td_agent, BaseAgent('O')]
    for _ in range(100):
        env.set_start_mark(start_mark)
        state = env.reset()
        while not env.done:
            _, mark = state
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            if agent == td_agent:
                action = agent.act(state, ava_actions, True)
            else:
                action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)

            if done:
                results.append(reward)
        start_mark = next_mark(start_mark)
    return np.mean(results)


def learn_td(epsilon, alpha, save_file=None):
    env = TicTacToe()
    agents = [TDAgent('O', epsilon, alpha),
              TDAgent('X', epsilon, alpha)]
    average_rewardsO = []
    average_rewardsX = []
    start_mark = 'O'
    for i in tqdm.tqdm(range(MAX_EPISODES)):
        episode = i+1
        # env.show_episode(False, episode)

        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            env.show_turn(False, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, next_state, reward)

            if done:
                set_state_value(state, reward)
                agent = agent_by_mark(agents, 'O')
                agent.epsilon = max(0, agent.epsilon - DECAY)

                agent = agent_by_mark(agents, 'X')
                agent.epsilon = max(0, agent.epsilon - DECAY)
            _, mark = state = next_state

        start_mark = next_mark(start_mark)

        if i % (MAX_EPISODES//100) == 0:
            # pass
            agent = agent_by_mark(agents, 'O')
            samp_valO = sample_game(agent)
            agent = agent_by_mark(agents, 'X')
            samp_valX = sample_game(agent)
            average_rewardsO.append(samp_valO)
            average_rewardsX.append(-1*samp_valX)
            plot(average_rewardsO, average_rewardsX)

    save_model(MODEL_FILE, V,
               MAX_EPISODES, EPSILON, ALPHA)
