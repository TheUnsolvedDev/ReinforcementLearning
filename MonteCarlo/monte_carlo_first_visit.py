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

    ax1.plot_surface(X, Y, state_values[:, :, 0],
                     cmap='viridis', edgecolor='none')
    ax2.plot_surface(X, Y, state_values[:, :, 1],
                     cmap='viridis', edgecolor='none')

    for ax in ax1, ax2:
        # ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')
    plt.show()


def first_visit_monte_carlo(env, episodes=1000):  # V = V_pi
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
            action = env.action_space.sample()
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
                state_values[S] += (total_reward)/state_count[S]
    return state_values


if __name__ == "__main__":
    episodes = 1000000*5

    state_values = first_visit_monte_carlo(env, episodes=episodes)
    # print(state_values)

    fig, axes = plt.subplots(nrows=2, figsize=(5, 8),
                             subplot_kw={'projection': '3d'})
    axes[0].set_title('value function without usable ace')
    axes[1].set_title('value function with usable ace')
    plot_blackjack(state_values, axes[0], axes[1])
