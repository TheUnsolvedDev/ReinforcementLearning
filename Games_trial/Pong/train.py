import numpy as np
import tensorflow as tf
import gym

from params import *
from models import *

import policy_gradient
import dqn

if __name__ == '__main__':
    policy_gradient.train(env)
    # dqn.train(env)
    
