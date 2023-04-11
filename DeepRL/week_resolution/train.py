import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from silence_tensorflow import silence_tensorflow
import argparse


from agents.params import *
from agents.models import *

from agents.reinforce import reinforce
from agents.reinforce_baseline import reinforce_baseline
from agents.dqn import DQN
from agents.actor_critic import NaiveActorCritic
from agents.doubledqn import DoubleDQN
from agents.duellingdqn import DuelingDQN

silence_tensorflow()
tf.random.set_seed(SEED)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    print('*** Training', env.unwrapped.spec.id, '***')
    print('Mode:', train_mode)
    if train_mode == 'reinforce':
        reinforce(env)
    elif train_mode == 'reinforce_baseline':
        reinforce_baseline(env)
    elif train_mode == 'DQN':
        DQN(env)
    elif train_mode == 'ActorCritic':
        NaiveActorCritic(env)
    elif train_mode == 'DoubleDQN':
        DoubleDQN(env)
    elif train_mode == 'DuelingDQN':
        DuelingDQN(env)
