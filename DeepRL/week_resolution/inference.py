import gymnasium as gym
import numpy as np
import sys

from agents.params import *
from agents.models import *


def simulate(env, num_games=10, model=None):
    for _ in range(num_games):
        state = env.reset()[0]
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            if model is None:
                action = env.action_space.sample()
            else:
                state = np.expand_dims(state, axis=0)
                logits = model(state, training=False)
                action = np.argmax(logits[0])
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            print('\rGame:', _, 'total_reward:',
                  total_reward, 'reward:', reward, end=' ')
            sys.stdout.flush()


if __name__ == '__main__':
    policy_model = model(in_dim, out_dim)
    policy_model.summary()
    policy_model.load_weights(
        'weights/'+env.unwrapped.spec.id+train_mode+'_model.h5')
    simulate(env=test_env, model=policy_model)
