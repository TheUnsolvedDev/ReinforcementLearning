

from agents import BaseAgent, HumanAgent, TDAgent, learn_td, load_model
from params import *
from env import TicTacToe, agent_by_mark, check_game_status, \
    next_state_show, code_to_mark, next_mark
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")


def play(agents, max_episode=100):
    start_mark = 'O'
    env = TicTacToe()
    results = []
    for _ in range(max_episode):
        env.set_start_mark(start_mark)
        state = env.reset()
        while not env.done:
            _, mark = state
            env.show_turn(True, mark)
            # time.sleep(0.25)
            env.render()

            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            try:
                action = agent.act(state, ava_actions)
            except TypeError:
                action = agent.act(ava_actions)
            state, reward, done, info = env.step(action)

            if done:
                results.append(reward)
        env.show_result(True, mark, reward)
        # rotate start
        start_mark = next_mark(start_mark)
    o_win = results.count(O_REWARD)
    x_win = results.count(X_REWARD)
    draw = len(results) - o_win - x_win
    print('O wins:', o_win, ',X wins:', x_win, ',Draw:', draw)


if __name__ == '__main__':
    # learn_td(EPSILON,ALPHA)
    V = load_model(MODEL_FILE)
    agents = [TDAgent('O', 0, 0, False, V), BaseAgent('X')]
    print(agents)
    play(agents)

    agents = [BaseAgent('O'), TDAgent('X', 0, 0, False, V)]
    print(agents)
    play(agents)

    agents = [TDAgent('O', 0, 0, False, V), TDAgent('X', 0, 0, False, V)]
    print(agents)
    play(agents)
    
    agents = [TDAgent('O', 0, 0, False, V), HumanAgent('X')]
    play(agents)
