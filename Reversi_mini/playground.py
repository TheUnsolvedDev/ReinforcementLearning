import matplotlib.pyplot as plt
import numpy as np
import time

from agents import BaseAgent, learn_td
from env import Reversi_env,mark_to_code,next_mark,code_to_mark,agent_by_mark
from params import *

def play(agents, max_episode=100):
    start_mark = 'B'
    env = Reversi_env(6)
    results = []
    for _ in range(max_episode):
        state = env.reset()
        while not env.terminated:
            _, mark = state
            env.render()
            agent = agent_by_mark(agents, mark)
            ava_actions = env.get_possible_actions()
            try:
                action = agent.act(state, ava_actions)
            except TypeError:
                action = agent.act(ava_actions)
            state, reward, done, info = env.step(action)


            if done:
                results.append(reward)
            time.sleep(0.1)
        # rotate start
        start_mark = next_mark(start_mark)
        print(env.winner,reward)
        input()

if __name__ == '__main__':
    learn_td(EPSILON,ALPHA)
    agents = [BaseAgent('B'), BaseAgent('W')]
    # play(agents)