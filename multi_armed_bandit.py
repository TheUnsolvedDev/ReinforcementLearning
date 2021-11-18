import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, probs):
        self.probs = probs  # success probabilities for each arm

    def step(self, action):
        # Pull arm and get stochastic reward (1 for success, 0 for failure)
        return 1 if (np.random.random() < self.probs[action]) else 0


class Agent:
    def __init__(self, actions, epsilon) -> None:
        self.actions = actions
        self.epsilon = epsilon

        self.n = np.zeros(self.actions, dtype=np.int)
        self.Q = np.zeros(self.actions, dtype=np.float32)

    def update_Q(self, action, reward):
        # Update Q action-value given (action, reward)
        self.n[action] += 1
        self.Q[action] += (1.0/self.n[action]) * (reward - self.Q[action])

    def get_action(self):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:  # explore
            return np.random.randint(self.actions)
        else:  # exploit
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))


def experiment(probs, N_episodes):
    env = Environment(probs)  # initialize arm probabilities
    agent = Agent(len(env.probs), epsilon=0.01)  # initialize agent
    actions, rewards = [], []
    for episode in range(N_episodes):
        action = agent.get_action()  # sample policy
        reward = env.step(action)  # take step + get reward
        agent.update_Q(action, reward)  # update Q
        actions.append(action)
        rewards.append(reward)
    return np.array(actions), np.array(rewards)


if __name__ == "__main__":
    probs = np.array([0.1, 0.5, 0.6, 0.8, 0.1, 0.25, 0.6, 0.45, 0.75, 0.65])
    # probs = np.array([0.5]*10)
    N_experiments = 10000
    N_steps = 500

    print('Running multi-armed bandit with {} actions, 0.01 epsilon.'.format(len(probs)))

    R = np.zeros((N_steps,))
    A = np.zeros((N_steps, len(probs)))

    for i in range(N_experiments):
        actions, rewards = experiment(probs, N_steps)
        if (i + 1) % (N_experiments / 100) == 0:
            print("[Experiment {}/{}] ".format(i + 1, N_experiments) +
                  "n_steps = {}, ".format(N_steps) +
                  "reward_avg = {}".format(np.sum(rewards) / len(rewards)))

        R += rewards

        for ind, elem in enumerate(actions):
            A[ind][elem] += 1
            
    R_avg = R / N_experiments
    
    plt.plot(R_avg,'.')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    
    plt.grid()
    ax = plt.gca()
    plt.xlim([1, N_steps])
    plt.show()
    
    for i in range(len(probs)):
        A_pct = 100 * A[:,i] / N_experiments
        steps = list(np.array(range(len(A_pct)))+1)
        plt.plot(steps, A_pct, "-",
                linewidth=4,
                label="Arm {} ({:.0f}%)".format(i+1, 100*probs[i]))
    plt.xlabel("Step")
    plt.ylabel("Count Percentage (%)")
    leg = plt.legend(loc='upper left', shadow=True)
    plt.xlim([1, N_steps])
    plt.ylim([0, 100])
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)
        
    plt.show()
