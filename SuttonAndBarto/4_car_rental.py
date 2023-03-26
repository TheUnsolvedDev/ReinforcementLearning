import numpy as np
from scipy.stats import poisson

import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns


params = {
        'max_cars': [20, 20],
        'max_moveable': 5,
        'revenue_from_rented': 10,
        'cost_for_moved': 2,
        'lambdas': np.array([3, 4, 3, 2])
        }


class CarRental:
    """ Example 4.2: Jack's Car Rental """

    def __init__(
            self,
            params=params,
            min_prob=1e-3
            ):
        self.params = params
        self.poisson_cache = {}

        self.state_maxes = [
                self.params['max_cars'][0],
                self.params['max_cars'][1],
                poisson.ppf(1-min_prob, self.params['lambdas'][0]),
                poisson.ppf(1-min_prob, self.params['lambdas'][1]),
                poisson.ppf(1-min_prob, self.params['lambdas'][2]),
                poisson.ppf(1-min_prob, self.params['lambdas'][3])
                ]

    def _get_poisson_logpmf(self, n, l):
        """ cache poisson pmf calls to save computation """
        if (n, l) not in self.poisson_cache:
            self.poisson_cache[(n, l)] = poisson.logpmf(n, l)
        return self.poisson_cache[(n, l)]

    def get_states(self):
        """ states are tuples of (car at 1, cars at 2) """
        return itertools.product(*[range(int(x) + 1) for x in self.state_maxes[:2]])

    def get_possible_actions(self, state):
        """ actions are cars moved from location 1 to location 2 (+ direction) or conversely (- direction)
        up to the max cars moveable - ie closed interval [-max_moveable, +max_moveable]
        up to the max cars available - ie close interval [-cars at loc 2, + cars at loc 1] """
        max_moveable = self.params['max_moveable']
        max_1, max_2, _, _, ret_1, ret_2 = self.state_maxes
        cars_1, cars_2 = state

        return np.arange(-min(min(max_moveable, cars_2), max_1 - cars_1),
                         +min(min(max_moveable, cars_1), max_2 - cars_2) + 1)

    def get_transition_state_revenue_prob(self, state, action):
        """ return a list of (next_state, prob) representing the state reachable
        from current state given action """

        if action not in self.get_possible_actions(state):
            raise 'Illegal action.'

        max_cars_1, max_cars_2, max_req_1, max_req_2, _, _ = self.state_maxes

        new_cars_1 = state[0] - action
        new_cars_2 = state[1] + action
        assert 0 <= new_cars_1 <= max_cars_1
        assert 0 <= new_cars_2 <= max_cars_2

        successors = []
        for i in range(int(max_req_1)):
            for j in range(int(max_req_2)):

                rented_1 = min(i, new_cars_1)
                rented_2 = min(j, new_cars_2)

                returned_1 = self.params['lambdas'][2]
                returned_2 = self.params['lambdas'][3]

                next_state_1 = min(new_cars_1 - rented_1 +
                                   returned_1, max_cars_1)
                next_state_2 = min(new_cars_2 - rented_2 +
                                   returned_2, max_cars_2)
                assert 0 <= next_state_1 <= max_cars_1
                assert 0 <= next_state_2 <= max_cars_2
                next_state = (next_state_1, next_state_2)

                joint_log_prob = 0
                joint_log_prob += self._get_poisson_logpmf(
                    i, self.params['lambdas'][0])
                joint_log_prob += self._get_poisson_logpmf(
                    j, self.params['lambdas'][1])
                joint_prob = np.exp(np.sum(joint_log_prob))

                revenue = (rented_1 + rented_2) * \
                           self.params['revenue_from_rented']

                successors.append((next_state, revenue, joint_prob))
        return successors


class PolicyIterationAgent:
    """ Ch4 p63 implementation of algorithm Policy Iteration (using iterative policy evaluation) """

    def __init__(self, mdp, discount=0.9, eps=1):
        self.mdp = mdp
        self.discount = discount
        self.eps = eps

        self.values = defaultdict(int)
        self.policy = defaultdict(int)

    def run_policy_iteration(self):
        """ perform a single cycle of policy iteration """
        while True:
            new_values = defaultdict(int)
            for state in self.mdp.get_states():
                action = self.policy[state]
                new_values[state] = self.compute_q_value(state, action)

            delta = np.sum([abs(self.values[k] - new_values[k])
                           for k, v in new_values.items()])
            print('Policy evaluation value delta: {}'.format(delta))
            if delta < self.eps:
                break

            self.values = new_values

        while True:
            new_policy = defaultdict(int)
            for state in self.mdp.get_states():
                q_values = {}
                for action in self.mdp.get_possible_actions(state):
                    q_values[(state, action)] = self.compute_q_value(
                        state, action)

                new_policy[state] = [
                    a for (s, a), v in q_values.items() if v == max(q_values.values())][0]

            if all([new_policy[k] == self.policy[k] for k in self.policy.keys()]):
                break

            self.policy = new_policy

    def get_policy(self):
        max_cars_1, max_cars_2 = self.mdp.state_maxes[:2]
        policy = np.zeros((max_cars_1 + 1, max_cars_2 + 1), dtype=np.int32)
        for k, v in self.policy.items():
            policy[k] = int(v)
        return policy

    def get_values(self):
        max_cars_1, max_cars_2 = self.mdp.state_maxes[:2]
        values = np.zeros((max_cars_1 + 1, max_cars_2 + 1), dtype=np.int32)
        for k, v in self.values.items():
            values[k] = int(v)
        return values

    def compute_q_value(self, state, action):

        q_value = - abs(action) * self.mdp.params['cost_for_moved']

        for (next_state, revenue, prob) in self.mdp.get_transition_state_revenue_prob(state, action):
            q_value += prob * (revenue + self.discount *
                               self.values[next_state])

        return q_value


def plot_policy(mdp, policy, ax, iter_count):
    sns.heatmap(
            policy,
            vmin=-mdp.params['max_moveable'],
            vmax=mdp.params['max_moveable'],
            yticklabels=list(range(mdp.params['max_cars'][0], -1, -1)),
            annot=True,
            ax=ax)
    ax.set_xlabel('cars at second location')
    ax.set_ylabel('cars at first location')
    ax.set_title(r'$\pi_{{%i}}$' % (iter_count))

def fig_4_2():
    mdp=CarRental()
    agent=PolicyIterationAgent(mdp)

    fig, axs=plt.subplots(2, 3)
    sns.set(rc={'font.size': 8, 'axes.labelsize': 'small',
            'xtick.labelsize': 'small', 'ytick.labelsize': 'small'})
    axs=axs.flatten()


    plot_policy(mdp, agent.get_policy(), axs[0], 0)


    for i in range(1, 5):
        print('Starting policy iteration cycle {}'.format(i))
        agent.run_policy_iteration()
        policy=agent.get_policy()
        plot_policy(mdp, np.flipud(policy), axs[i], i)


    axs[-1].axis('off')
    ax=fig.add_subplot(2, 3, 6, projection='3d')
    x=np.arange(mdp.params['max_cars'][1] + 1)
    y=np.arange(mdp.params['max_cars'][0] + 1)[::-1]
    xx, yy=np.meshgrid(x, y)
    ax.plot_wireframe(xx, yy, np.flipud(agent.get_values()))
    ax.set_xlabel('# cars at second location', fontsize=8)
    ax.set_ylabel('# cars at first location', fontsize=8)
    ax.set_title(r'$v_{{%i}}$' % i)

    plt.tight_layout()
    plt.savefig('figures/ch04_4_2.png')
    plt.close()

if __name__ == '__main__':
    fig_4_2()
