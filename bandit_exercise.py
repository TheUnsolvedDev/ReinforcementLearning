import numpy as np


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
        reward = self.bandits[choice - 1].lever()
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
    bandits = [GaussianBandit(mean=0, stdev=1),
               GaussianBandit(mean=1, stdev=1),
               GaussianBandit(mean=2, stdev=1)]
    game = Game(bandits)
    game.match()
