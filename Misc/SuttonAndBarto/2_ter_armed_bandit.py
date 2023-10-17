import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_fig(name, ch='ch02'):
    plt.savefig('figures/{}_{}.png'.format(ch, name))
    plt.close()


class BaseBandit:
    def __init__(self, k_arm=10, eps=0, initial_q=0, true_q_mean=0):
        self.k_arm = k_arm
        self.possible_actions = np.arange(self.k_arm)
        self.eps = eps
        self.initial_q = initial_q
        self.true_q_mean = true_q_mean

        self.reset()

    def reset(self):
        self.q_true = np.random.randn(self.k_arm) + self.true_q_mean
        self.q_estimate = np.zeros(self.k_arm) + self.initial_q
        self.action_count = np.zeros(self.k_arm)
        self.optimal_action_freq = 0

    def act(self):
        if np.random.rand() < self.eps:
            action = np.random.choice(self.possible_actions)
        else:
            action = np.argmax(self.q_estimate)
        return action

    def reward(self, action_idx):
        return np.random.randn() + self.q_true[action_idx]

    def update_q(self, action, reward):
        self.q_estimate[action] += 1/self.action_count[action] * \
            (reward - self.q_estimate[action])

    def step(self):
        action = self.act()
        reward = self.reward(action)
        self.action_count[action] += 1
        self.update_q(action, reward)

        if action == np.argmax(self.q_true):
            self.optimal_action_freq += 1 / \
                np.sum(self.action_count) * (1 - self.optimal_action_freq)

        return action, reward


class ExponentialAverageBandit(BaseBandit):
    def __init__( self, step_size=0.1, **kwargs ):
        super().__init__(**kwargs)
        self.step_size = step_size

    def update_q(self, action, reward):
        self.q_estimate[action] += self.step_size * \
            (reward - self.q_estimate[action])


class UCBBandit(BaseBandit):
    def __init__( self,c=2, **kwargs):
        super().__init__(**kwargs)
        self.c = c

    def act(self):
        if np.random.rand() < self.eps:
            action = np.random.choice(self.possible_actions)
        else:
            t = np.sum(self.action_count) + 1
            q = self.q_estimate + self.c * \
                np.sqrt(np.log(t) / (self.action_count + 1e-6))
            action = np.argmax(q)
        return action


class GradientBandit(BaseBandit):
    def __init__(self, baseline=True, step_size=0.1, **kwargs):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.step_size = step_size
        self.average_reward = 0

    def act(self):
        e = np.exp(self.q_estimate)
        self.softmax = e / np.sum(e)
        return np.random.choice(self.possible_actions, p=self.softmax)

    def update_q(self, action, reward):

        self.average_reward += 1 / \
            np.sum(self.action_count) * (reward - self.average_reward)
        baseline = self.average_reward if self.baseline else 0

        mask = np.zeros_like(self.softmax)
        mask[action] = 1
        self.q_estimate += self.step_size * \
            (reward - baseline) * (mask - self.softmax)


def run_bandits(bandits, n_runs, n_steps):
    rewards = np.zeros((len(bandits), n_runs, n_steps))
    optimal_action_freqs = np.zeros_like(rewards)

    for b, bandit in enumerate(bandits):
        for run in tqdm(range(n_runs)):
            bandit.reset()
            for step in range(n_steps):
                action, reward = bandit.step()
                rewards[b, run, step] = reward
                if action == np.argmax(bandit.q_true):
                    optimal_action_freqs[b, run, step] = 1

    avg_rewards = rewards.mean(axis=1)
    avg_optimal_action_freqs = optimal_action_freqs.mean(axis=1)

    return avg_rewards, avg_optimal_action_freqs


def fig_2_1():
    plt.violinplot(np.random.randn(200, 10) +
                   np.random.randn(10), showmeans=True)
    plt.xticks(np.arange(1, 11), np.arange(1, 11))
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    save_fig('fig_2_1')


def fig_2_2(runs=2000, steps=1000, epsilons=[0, 0.01, 0.1]):
    bandits = [BaseBandit(eps=eps) for eps in epsilons]
    avg_rewards, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, avg_rewards):
        plt.plot(rewards, label=r'$\epsilon$ = {}'.format(eps), lw=1)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, optimal_action_freq in zip(epsilons, avg_optimal_action_freqs):
        plt.plot(optimal_action_freq,
                 label=r'$\epsilon$ = {}'.format(eps), lw=1)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.tight_layout()
    save_fig('fig_2_2')


def fig_2_3(runs=2000, steps=1000, epsilons=[0, 0.1], initial_qs=[5, 0]):
    bandits = []
    for eps, initial_q in zip(epsilons, initial_qs):
        bandits.append(ExponentialAverageBandit(
            eps=eps, initial_q=initial_q, step_size=0.1))

    _, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    for i, (eps, initial_q) in enumerate(zip(epsilons, initial_qs)):
        plt.plot(avg_optimal_action_freqs[i], label=r'Q1 = {}, $\epsilon$ = {}'.format(
            initial_q, eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    save_fig('fig_2_3')


def fig_2_4(runs=2000, steps=1000):
    bandits = [UCBBandit(eps=0, c=2), BaseBandit(eps=0.1)]

    _, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    plt.plot(avg_optimal_action_freqs[0],
             label='UCB c = {}'.format(bandits[0].c))
    plt.plot(
        avg_optimal_action_freqs[1], label=r'$\epsilon$-greedy $\epsilon$ = {}'.format(bandits[1].eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    save_fig('fig_2_4')


def fig_2_5(runs=2000, steps=1000):
    bandits = [
        GradientBandit(step_size=0.1, true_q_mean=4, baseline=True),
        GradientBandit(step_size=0.4, true_q_mean=4, baseline=True),
        GradientBandit(step_size=0.1, true_q_mean=4, baseline=False),
        GradientBandit(step_size=0.4, true_q_mean=4, baseline=False)]

    _, avg_optimal_action_freqs = run_bandits(bandits, runs, steps)

    for i, bandit in enumerate(bandits):
        plt.plot(avg_optimal_action_freqs[i],
                 label='step_size = {}, baseline = {}'.format(bandit.step_size, bandit.baseline))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    save_fig('fig_2_5')


if __name__ == '__main__':
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_2_4()
    fig_2_5()
