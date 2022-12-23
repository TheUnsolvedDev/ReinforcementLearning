import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import tqdm

from params import *
from models import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model = lenet5()
model.summary()
model.load_weights('lenet_PG.h5')


def select_action(state, epsilon=EPSILON):
    if tf.random.uniform(()) < epsilon:
        return tf.random.uniform((), minval=0, maxval=env.action_space.n, dtype=tf.int64)
    else:
        logits = model(tf.expand_dims(state, 0))
        return tf.argmax(logits, axis=1)[0]


def discount_rewards(rewards):
    discounted_rewards = np.zeros(len(rewards), dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0.0
        running_add = running_add * GAMMA + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


@tf.function
def train_step(states, actions, rewards):
    with tf.GradientTape() as tape:
        action_probs = model(states)
        neg_log_probs = -tf.math.log(action_probs)
        action_mask = tf.one_hot(actions, depth=env.action_space.n)
        action_neg_log_probs = tf.reduce_sum(
            action_mask * neg_log_probs, axis=1)
        loss = tf.reduce_mean(rewards * action_neg_log_probs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def plotting(scores, mean_scores):
    plt.ion()
    plt.clf()
    plt.title('Training Reinforce.')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('TrainingCurve.png')


def train(env, episodes=EPISODES, plot=False):
    total_rewards = []
    mean_rewards = []

    for episode in tqdm.tqdm(range(EPISODES)):
        state = env.reset()
        state = preprocess(state[0])

        states, actions, rewards = [], [], []
        done = False
        total_reward = 0
        while not done:
            action = select_action(state)
            next_state, reward, done, _, truncated = env.step(action)
            total_reward += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = preprocess(next_state)

        states = tf.stack(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = discount_rewards(rewards)
        train_step(states, actions, rewards)
        print('The Total Reward @episodes:', episode, 'is',
              total_reward, 'mean reward', np.mean(total_rewards))

        if (not episode % 100):
            total_rewards.append(total_reward)
            mean = np.mean(total_rewards)
            mean_rewards.append(mean)

            if plot:
                plotting(total_rewards, mean_rewards)

            if mean >= total_reward:
                model.save_weights('lenet_PG.h5')
