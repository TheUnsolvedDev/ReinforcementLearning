import tensorflow as tf
import gym
import cv2
import numpy as np
import tqdm
from collections import defaultdict
import tqdm

from params import *
from replay_memory import MemoryBuffer
from model import cnn_model

trial_env = gym.make('PongNoFrameskip-v4', render_mode='human')
train_env = gym.make('PongNoFrameskip-v4')
n_actions = train_env.action_space.n


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def tolist(a):
    try:
        return list(tolist(i) for i in a)
    except TypeError:
        return a


def pre_preprocess(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray_image, (IMAGE_SIZE[0], IMAGE_SIZE[1]))


def random_gameplay(num_games=NUM_TRIAL_GAMES, is_random=True):
    mod = cnn_model()
    try:
        mod.load_weights('cnn_mod.h5')
    except FileNotFoundError:
        model.save_weights('cnn_mod.h5')

    for _ in range(num_games):
        total_reward = 0
        states = [pre_preprocess(trial_env.reset()) for i in range(4)]
        done = False
        while not done:
            if is_random:
                action = trial_env.action_space.sample()
            else:
                states_np = np.array(states).astype(int).reshape(IMAGE_SIZE)
                action = np.argmax(mod(np.expand_dims(states_np, axis=0)))
            temp_state, reward, done, info = trial_env.step(action)
            states.pop(0)
            states += [pre_preprocess(temp_state)]
            total_reward += reward

        print("Reward: ", total_reward, "Episodes: ", _)


def monte_carlo(env, episodes=NUM_TRAIN_GAMES):
    Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))
    state_count = defaultdict(lambda: np.zeros(n_actions, dtype=int))
    epsilon = EPS_START
    decay = EPS_DECAY
    mod = cnn_model()
    buffer = MemoryBuffer()

    for _ in tqdm.tqdm(range(episodes)):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='cnn_mod.h5', save_weights_only=True, monitor='loss', save_best_only=True),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs_total/logs'+str(_), histogram_freq=1, write_graph=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
        ]

        total_reward = 0
        state_actions = []
        rewards = []

        states = [pre_preprocess(env.reset()) for i in range(4)]
        done = False
        while not done:
            if np.random.randn() <= epsilon:
                action = env.action_space.sample()
            else:
                states_np = np.array(states).astype(int).reshape(IMAGE_SIZE)
                action = np.argmax(mod(np.expand_dims(states_np, axis=0)))
            state_actions.append((states, action))
            temp_state, reward, done, info = env.step(action)
            states.pop(0)
            states += [pre_preprocess(temp_state)]
            rewards.append(reward)

        for ind in tqdm.tqdm(range(len(state_actions)-1, -1, -1)):
            S, A = state_actions[ind]
            R = rewards[ind]
            total_reward += R

            S = totuple(S)
            state_count[S][A] += 1
            Q[S][A] += (total_reward - Q[S][A])/state_count[S][A]

            S_np = np.array(tolist(S)).reshape(IMAGE_SIZE)
            buffer.push(S_np, Q[S])

        if _ % 5 == 0:
            full_data = buffer.sample()
            data = full_data[0]
            labels = full_data[1]
            mod.compile(loss='mse', optimizer='adam')
            mod.fit(data, labels, batch_size=64,
                    epochs=100, callbacks=callbacks)
            mod.save_weights('cnn_mod.h5')
            random_gameplay(num_games=1, is_random=False)

        if _ % 10 == 0:
            Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))
            state_count = defaultdict(lambda: np.zeros(n_actions, dtype=int))
            epsilon *= decay
            epsilon = max(epsilon, EPS_END)

    return Q


if __name__ == '__main__':
    # random_gameplay(is_random=False)
    monte_carlo(train_env)
