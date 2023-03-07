import gymnasium as gym
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


def random_policy(nA):
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A
    return policy_fn


def off_policy_monte_carlo_control(env, episodes=1000):  # Pi = Pi*
    Q = defaultdict(lambda: np.zeros(n_actions))
    count = defaultdict(lambda: np.zeros(n_actions))
    target_policy = defaultdict(int)

    for ep in tqdm.tqdm(range(episodes)):
        total_reward = 0
        states = []
        actions = []
        rewards = []

        state = env.reset()[0]
        done = False
        while not done:
            states.append(state)
            probs = random_policy(n_actions)(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            actions.append(action)
            state, reward, done, info, truncated = env.step(action)
            rewards.append(reward)

        W = 1.0
        for ind in range(len(states)-1, -1, -1):
            S = states[ind]
            A = actions[ind]
            R = rewards[ind]

            total_reward += R
            count[S][A] += W
            Q[S][A] += W*(total_reward - Q[S][A])/count[S][A]
            target_policy[S] = np.argmax([Q[S][a] for a in range(n_actions)])

            if A != target_policy[S]:
                break
            W *= 1/(random_policy(n_actions))(S)[A]

    return Q, target_policy


def games_trial(env, policy, no_of_games=1000):
    count = 0
    for games in tqdm.tqdm(range(no_of_games)):
        done = False
        state = env.reset()[0]
        while not done:
            action = policy[state]
            state, reward, done, info, truncated = env.step(action)

            if reward == 1:
                count += 1

    print(" No of games won:", count, "out of", no_of_games)


if __name__ == '__main__':
    episodes = 1000000

    Q, policy = off_policy_monte_carlo_control(env, episodes=episodes)
    # print(policy)

    fig, axes = plt.subplots(nrows=2, figsize=(5, 8))
    axes[0].set_title('Policy without usable ace')
    axes[1].set_title('Policy function with usable ace')

    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros(
        (len(player_sum), len(dealer_show), len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = policy[player, dealer, ace]

    X, Y = np.meshgrid(dealer_show, player_sum)

    axes[0].scatter(X, Y, c=state_values[:, :, 0],
                    cmap='viridis', edgecolor='none')
    axes[1].scatter(X, Y, c=state_values[:, :, 1],
                    cmap='viridis', edgecolor='none')

    for ax in [axes[0], axes[1]]:
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
    plt.show()

    games_trial(env, policy, no_of_games=100)

    env.close()
