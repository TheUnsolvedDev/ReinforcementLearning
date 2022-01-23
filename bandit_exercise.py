import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GaussianBandit:
    def __init__(self, mean=0, stdev=1) -> None:
        self.mean = mean
        self.stdev = stdev

    def lever(self):
        return np.round(np.random.normal(self.mean, self.stdev))


class BernoulliBandit(object):
    def __init__(self, p):
        self.p = p

    def display_ad(self):
        reward = np.random.binomial(n=1, p=self.p)
        return reward


class Game:
    def __init__(self, bandits) -> None:
        self.bandits = bandits
        np.random.shuffle(self.bandits)
        self.reset_game()

    def play(self, choice):
        reward = self.bandits[choice - 1].display_ad()
        self.rewards.append(reward)
        self.total_reward += reward
        self.n_played += 1
        return reward

    def match(self):
        self.reset_game()
        print("Game started. Enter 0 as input to end the game.")
        while True:
            print(f"\n -- Round {self.n_played}")
            choice = int(input(f"Choose a machine " +
                         f"from 1 to {len(self.bandits)}: "))
            if choice in range(1, len(self.bandits) + 1):
                reward = self.play(choice)
                print(f"Machine {choice} gave " + f"a reward of {reward}.")
                avg_rew = self.total_reward/self.n_played
                print(f"Your average reward " + f"so far is {avg_rew}.")
            else:
                break
        print("Game has ended.")
        if self.n_played > 0:
            print(f"Total reward is {self.total_reward}" +
                  f" after {self.n_played} round(s).")
            avg_rew = self.total_reward/self.n_played
            print(f"Average reward is {avg_rew}.")

    def reset_game(self):
        self.rewards = []
        self.total_reward = 0
        self.n_played = 0


if __name__ == '__main__':
    adA = BernoulliBandit(0.004)
    adB = BernoulliBandit(0.016)
    adC = BernoulliBandit(0.02)
    adD = BernoulliBandit(0.028)
    adE = BernoulliBandit(0.031)
    ads = [adA, adB, adC, adD, adE]
    game = Game(ads)
    # game.match()

    n_test = 10000
    n_prod = 100000
    n_ads = len(ads)
    Q = np.zeros(n_ads)  # Q, action values
    N = np.zeros(n_ads)  # N, total impressions
    total_reward = 0
    avg_rewards = []  # Save average rewards over time

    for i in range(n_test):
        ad_chosen = np.random.randint(n_ads)
        R = ads[ad_chosen].display_ad()  # Observe reward
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)

    best_ad_index = np.argmax(Q)
    print("The best performing ad is {}".format(chr(ord('A') + best_ad_index)))

    plt.plot(avg_rewards)
    plt.xlabel("Number of rounds")
    plt.ylabel("Average reward for testings")
    plt.show()

    total_reward = 0
    avg_rewards = []  # Save average rewards over time

    for i in range(n_prod):
        ad_chosen = best_ad_index
        R = ads[ad_chosen].display_ad()  # Observe reward
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)

    plt.plot(avg_rewards)
    plt.xlabel("Number of rounds")
    plt.ylabel("Average reward for prods")
    plt.show()

    eps = 0.1
    n_prod = 100000
    n_ads = len(ads)
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_rewards = []

    ad_chosen = np.random.randint(n_ads)
    for i in range(n_prod):
        R = ads[ad_chosen].display_ad()
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)
        # Select the next ad to display
        if np.random.uniform() <= eps:
            ad_chosen = np.random.randint(n_ads)
        else:
            ad_chosen = np.argmax(Q)

    best_ad_index = np.argmax(Q)
    print("The best performing ad is {}".format(chr(ord('A') + best_ad_index)))

    plt.plot(avg_rewards)
    plt.xlabel("Number of rounds")
    plt.ylabel("Average reward for testings")
    plt.show()

    total_reward = 0
    avg_rewards = []  # Save average rewards over time

    for i in range(n_prod):
        ad_chosen = best_ad_index
        R = ads[ad_chosen].display_ad()  # Observe reward
        N[ad_chosen] += 1
        Q[ad_chosen] += (1 / N[ad_chosen]) * (R - Q[ad_chosen])
        total_reward += R
        avg_reward_so_far = total_reward / (i + 1)
        avg_rewards.append(avg_reward_so_far)

    plt.plot(avg_rewards)
    plt.xlabel("Number of rounds")
    plt.ylabel("Average reward for prods")
    plt.show()
