import numpy as np
import matplotlib.pyplot as plt

# GaussianBandit
'''
	consider a situation where we choose from 4 slot machine that gives
	the maximum output among all the 4 slots. The 4 slots
	are biased under one condition (they are randomly distributed)
	in a normal distribution having various values of mean,std for our
	study purpose, futhur more this distribution can be changed
	according to your specific need. Considering our given scenerios
	the above mention situations prevails.

	Gaussian Slot1(1,1)
	Gaussian Slot2(2,3)
	Gaussian Slot3(3,1)
	Gaussian Slot4(1,3)
'''


class GaussBandit:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def pull_lever(self):
        return np.random.normal(loc=self.mean, scale=self.std)


class MAB:
    def __init__(self, n_bandits, bandit_l, random):
        self.n_bandits = n_bandits
        self.bandit_l = bandit_l
        # for increasing the randomnes
        if random:
            np.random.shuffle(self.bandit_l)

        self.reset_game()

    def reset_game(self):
        self.avg_reward = 0
        self.rewards = []
        self.total_reward = 0
        self.n_games = 0

    def game_user(self):
        self.reset_game()
        print('Multi arm bandit')
        print('NB : 0 to cancel')
        while True:
            n = int(input('Select between 1 - 4: '))
            if n == 0:
                print('Game Ended')
                break
            elif n >= 1 and n <= 4:
                choice = self.bandit_l[n-1]
                reward = choice.pull_lever()
                self.n_games += 1
                self.rewards.append(reward)
                self.total_reward = sum(self.rewards)
                self.avg_reward = self.total_reward/self.n_games
                print()
                print('Reward on', n, ':', reward)
                print('Avg reward on', self.n_games, ':', self.avg_reward)
                print()

    def game_comp_AB_test(self):
        self.reset_game()
        train_size = 10000
        test_size = 1000

        plt.figure(figsize=(14, 4))
        Q = np.zeros(self.n_bandits)
        N = np.zeros(self.n_bandits)

        total_reward = 0
        avg_reward = []
        for i in range(train_size):
            choice = np.random.randint(self.n_bandits)
            R = self.bandit_l[choice].pull_lever()
            N[choice] += 1
            Q[choice] += (1/N[choice])*(R-Q[choice])
            total_reward += R
            avg_reward.append(total_reward/(i+1))

        plt.subplot(121)
        plt.plot(avg_reward)
        plt.title('Average Reward for training')
        plt.xlabel('rounds')
        plt.ylabel('avg_rewards')

        best = np.argmax(Q)
        print('Best one is :', best+1)
        print('Reward approx train:', avg_reward[-1])

        total_reward = 0
        avg_reward = []
        for i in range(test_size):
            choice = best
            R = self.bandit_l[choice].pull_lever()
            total_reward += R
            avg_reward.append(total_reward/(i+1))

        print('Reward approx test:', avg_reward[-1])

        plt.subplot(122)
        plt.plot(avg_reward)
        plt.title('Average Reward under best bandit')
        plt.xlabel('rounds')
        plt.ylabel('avg_rewards')

        plt.show()

    def game_comp_Epsil_greed(self,eps = 0.1):
        self.reset_game()
        train_size = 10000
        test_size = 1000

        plt.figure(figsize=(14, 4))
        Q = np.zeros(self.n_bandits)
        N = np.zeros(self.n_bandits)

        total_reward = 0
        avg_reward = []
        for i in range(train_size):
            if np.random.uniform() <= eps:
                choice = np.random.randint(self.n_bandits)
            else:
                choice = np.argmax(Q)
            R = self.bandit_l[choice].pull_lever()
            N[choice] += 1
            Q[choice] += (1/N[choice])*(R-Q[choice])
            total_reward += R
            avg_reward.append(total_reward/(i+1))

        plt.subplot(121)
        plt.plot(avg_reward)
        plt.title('Average Reward for training')
        plt.xlabel('rounds')
        plt.ylabel('avg_rewards')

        best = np.argmax(Q)
        print('Best one is :', best+1)
        print('Reward approx train:', avg_reward[-1])

        total_reward = 0
        avg_reward = []
        for i in range(test_size):
            choice = best
            R = self.bandit_l[choice].pull_lever()
            total_reward += R
            avg_reward.append(total_reward/(i+1))

        print('Reward approx test:', avg_reward[-1])

        plt.subplot(122)
        plt.plot(avg_reward)
        plt.title('Average Reward under best bandit')
        plt.xlabel('rounds')
        plt.ylabel('avg_rewards')

        plt.show()
        
        return avg_reward


if __name__ == '__main__':
    gb1 = GaussBandit(1, 1)
    gb2 = GaussBandit(2, 3)
    gb3 = GaussBandit(3, 1)
    gb4 = GaussBandit(1, 3)
    slot_list = [gb1, gb2, gb3, gb4]

    mab = MAB(4, slot_list, False)
    # mab.game_user()
    mab.game_comp_AB_test()
    mab.game_comp_Epsil_greed()