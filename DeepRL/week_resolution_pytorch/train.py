from agents.params import *
from agents.models import *
from agents.dqn import DQN


if __name__ == '__main__':
    print('*** Training', env.unwrapped.spec.id, '***')
    print('Mode:', train_mode)
    if train_mode == 'DQN':
        DQN(env)