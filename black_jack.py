"""
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.
    
    ### Description
    
    Card Values:
    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.
    
    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.
    
    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.
    
    ### Action Space
    
    There are two actions: stick (0), and hit (1).
    
    ### Observation Space
    
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).
    
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).
    
    ### Rewards
    
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:
        +1.5 (if <a href="#nat">natural</a> is True)
        +1 (if <a href="#nat">natural</a> is False)
        
    ### Arguments
    
    ```
    gym.make('Blackjack-v1', natural=False, sab=False)
    ```
    <a id="nat">`natural=False`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).
    
    <a id="sab">`sab=False`</a>: Whether to follow the exact rules outlined in the book by
    Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
    If the player achieves a natural blackjack and the dealer does not, the player
    will win (i.e. get a reward of +1). The reverse rule does not apply.
    If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).
    
    ### Version History
    * v0: Initial versions release (1.0.0)
"""

import re
import gym
import numpy as np
import time
import sys
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

env = gym.make('Blackjack-v1')
env = env.unwrapped

n_actions = env.action_space.n
n_observations = env.observation_space


def sample_policy(state):
    score, dealer_card, usable_ace = state
    return 0 if score >= 20 else 1


def plot_blackjack(V, ax1, ax2):
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros(
        (len(player_sum), len(dealer_show), len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]

    X, Y = np.meshgrid(player_sum, dealer_show)

    ax2.scatter(X, Y, state_values[:, :, 0],
                     cmap='viridis', edgecolor='none')
    ax1.scatter(X, Y, state_values[:, :, 1],
                     cmap='viridis', edgecolor='none')

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')
    plt.show()


def first_visit_monte_carlo(env, episodes=1000):
    state_values = defaultdict(float)
    state_count = defaultdict(int)

    for ep in tqdm.tqdm(range(episodes)):
        states = []
        actions = []
        rewards = []

        done = False
        state = env.reset()
        while not done:
            states.append(state)
            action = sample_policy(state)
            actions.append(action)
            state, reward, done, info = env.step(action)
            rewards.append(reward)

        total_reward = 0
        for ind in range(len(states)-1, -1, -1):
            S = states[ind]
            R = rewards[ind]

            total_reward += R
            if S not in states[:ind]:
                state_count[S] += 1
                state_values[S] += (total_reward -
                                    state_values[S])/state_count[S]
    return state_values

def simulate_env(env,policy,is_first = True):
    states = []
    actions = []
    rewards = []
    
    state = env.reset()
    done = False
    while not done:
        states.append(state)
        if is_first:
            action = sample_policy(state)
        else:
            action = policy[state]
        actions.append(action)
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        
    return states,actions,rewards,policy


def monte_carlo_exploring(env, episodes=1000):
    Q = {}
    for player_sum in range(0, 22):
        for dealer_card in range(1, 11):
            for usable_ace in [True, False]:
                state = (player_sum, dealer_card, usable_ace)
                Q[state] = {0: 0, 1: 0}    
                
    policy = defaultdict(int)
    state_count = defaultdict(int)

    is_first = True
    for ep in tqdm.tqdm(range(episodes)):
        total_reward = 0
        states,actions,rewards,policy = simulate_env(env,policy, is_first)

        for ind in range(len(states)-1, -1, -1):
            S = states[ind]
            R = rewards[ind]
            A = actions[ind]

            total_reward += R
            if S not in states[:ind]:
                state_count[S] += 1
                Q[S][A] += (total_reward - Q[S][A])/state_count[S]
                policy[S] = np.argmax([Q[S][a] for a in range(n_actions)])
        is_first = False
    return Q, policy


if __name__ == '__main__':
    episodes = 500000

    # state_values = first_visit_monte_carlo(env, episodes=episodes)
    # print(state_values)

    # fig, axes = plt.subplots(nrows=2, figsize=(5, 8),
    #                          subplot_kw={'projection': '3d'})
    # axes[0].set_title('value function without usable ace')
    # axes[1].set_title('value function with usable ace')
    # plot_blackjack(state_values, axes[0], axes[1])

    Q, policy = monte_carlo_exploring(env, episodes=episodes)
    print(policy)

    fig, axes = plt.subplots(nrows=2, figsize=(5, 8),
                             subplot_kw={'projection': '3d'})
    axes[0].set_title('Policy without usable ace')
    axes[1].set_title('Policy function with usable ace')
    plot_blackjack(policy, axes[0], axes[1])

    env.close()