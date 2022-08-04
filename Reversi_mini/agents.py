import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import tqdm
import json

from params import *
from env import Reversi_env, mark_to_code, next_mark, code_to_mark, next_state_show, agent_by_mark

plt.ion()

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
    
def sample_game(td_agent):
    start_mark = 'B'
    env = Reversi_env()
    results = []
    if td_agent.mark == 'B':
        agents = [td_agent, BaseAgent('W')]
    else:
        agents = [td_agent, BaseAgent('B')]
    for _ in range(100):
        state = env.reset()
        while not env.terminated:
            _, mark = state
            agent = agent_by_mark(agents, mark)
            ava_actions = env.get_possible_actions()
            if agent == td_agent:
                action = agent.act(state, ava_actions, True)
            else:
                action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)

            if done:
                results.append(reward)
        start_mark = next_mark(start_mark)
    return np.mean(results)


class BaseAgent:
    def __init__(self, mark) -> None:
        self.mark = mark

    def act(self, state, actions):
        return np.random.choice(actions)


V = {}
visits = defaultdict(int)


def set_state_value(state, value):
    visits[state] += 1
    V[state] = value


class QAgent:
    def __init__(self, mark, epsilon, alpha, train=True, V=None) -> None:
        self.mark = mark
        self.epsilon = epsilon
        self.alpha = alpha
        self.epsilon = epsilon
        self.V = V

    def act(self, state, actions, full_greedy=False):
        if self.V:
            if self.mark == 'W':
                ava_values = [self.V[next_state_show(
                    state, action)] for action in actions]
                best = np.max(ava_values)
            else:
                ava_values = [self.V[next_state_show(
                    state, action)] for action in actions]
            indices = [i for i, v in enumerate(ava_values) if v == best]
            aidx = np.random.choice(indices)
            action = actions[aidx]
            return action
        return self.greedy(state, actions, full_greedy)

    def greedy(self, state, ava_actions, full_greedy=False):

        e = np.random.uniform(low=0, high=self.epsilon)
        if e < self.epsilon:
            action = self.random_action(ava_actions)
        else:
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

        if self.mark == 'W':
            best = np.max(ava_values)
        if self.mark == 'B':
            best = np.min(ava_values)
        indices = [i for i, v in enumerate(ava_values) if v == best]
        aidx = np.random.choice(indices)
        action = ava_actions[aidx]
        return action

    def retake_val(self, state):
        return V[state]

    def learn(self, state, next_state, reward):
        if state not in V:
            set_state_value(state,reward)
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


def learn_td(epsilon, alpha, save_file=None):
    env = Reversi_env()
    agents = [QAgent('B', epsilon, alpha),
              QAgent('W', epsilon, alpha)]
    average_rewardsO = []
    average_rewardsX = []
    start_mark = 'B'
    for i in tqdm.tqdm(range(MAX_EPISODES)):
        episode = i+1
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.get_possible_actions()
            action = agent.act(state, ava_actions)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, next_state, reward)

            if done:
                set_state_value(state, reward)
                agent = agent_by_mark(agents, 'B')
                agent.epsilon = max(0, agent.epsilon - DECAY)

                agent = agent_by_mark(agents, 'W')
                agent.epsilon = max(0, agent.epsilon - DECAY)
            _, mark = state = next_state

        start_mark = next_mark(start_mark)

        if i % (MAX_EPISODES//100) == 0:
            # pass
            agent = agent_by_mark(agents, 'B')
            samp_valO = sample_game(agent)
            agent = agent_by_mark(agents, 'W')
            samp_valX = sample_game(agent)
            average_rewardsO.append(samp_valO)
            average_rewardsX.append(-1*samp_valX)
            plot(average_rewardsO, average_rewardsX)

    save_model(MODEL_FILE, V,
               MAX_EPISODES, EPSILON, ALPHA)


if __name__ == '__main__':
    pass
