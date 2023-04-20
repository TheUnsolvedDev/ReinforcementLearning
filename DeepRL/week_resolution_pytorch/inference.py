import gymnasium as gym
import numpy as np
import sys
import matplotlib.pyplot as plt

from agents.params import *
from agents.models import *
from agents.utils import *


@tf.function
def act(actor, state):
    actor.layers[-1].activation = tf.keras.activations.softmax
    prob = actor(tf.expand_dims(state, axis=0), training=False)
    logits = tf.math.log(prob)
    action = tf.random.categorical(logits, 1)
    return int(action[0, 0])


def simulate(env, num_games=10, model=None):
    for _ in range(num_games):
        state = env.reset()[0]
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            # image_datas = state_frames.reshape((4, 84, 84))
            # f, axarr = plt.subplots(2, 2)
            # axarr[0, 0].imshow(image_datas[0])
            # axarr[0, 1].imshow(image_datas[1])
            # axarr[1, 0].imshow(image_datas[2])
            # axarr[1, 1].imshow(image_datas[3])
            # plt.show()

            if model is None:
                action = env.action_space.sample()
            else:
                action = act(model, np.array(state).T).numpy()
            print(action)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            print('\rGame:', _, 'total_reward:',
                  total_reward, 'reward:', reward, end=' ')
            sys.stdout.flush()


if __name__ == '__main__':
    print(env.unwrapped.spec.id)
    test_env = make_env(env.unwrapped.spec.id, render_mode='human')

    policy_model = model(out_dim)
    policy_model.summary()
    policy_model.load_weights(
        'weights/'+env.unwrapped.spec.id+train_mode+'_model.h5')
    simulate(env=test_env, model=policy_model)
