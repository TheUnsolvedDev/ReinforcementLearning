import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from params import *
from models import *
from utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def act(actor, state):
    prob = actor(np.array([state]))
    prob = prob.numpy()
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    action = dist.sample()
    return int(action.numpy()[0])


def reinforce(env):
    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
    policy_model = model(in_dim, out_dim)
    policy_model.layers[-1].activation = tf.keras.activations.softmax
    policy_model.summary()
    log_dir = os.path.join(
        "logs_reinforce", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    episodic_improvements = []

    for episode in range(NUM_EPISODES):
        all_states, all_actions, all_rewards = [], [], []
        for i in range(NUM_TRAJECTORIES):
            states, actions, rewards = [], [], []
            state = env.reset()[0]
            done = False
            truncated = False
            while not (done or truncated):
                states.append(state)
                action = act(policy_model,state)
                actions.append(action)
                state, reward, done, truncated, info = env.step(
                    action)
                rewards.append(reward)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(discounted_rewards(rewards))
        with tf.GradientTape() as tape:
            policy_loss = 0
            for i in range(len(all_states)):
                state = np.array([all_states[i]])
                action = all_actions[i]
                reward = all_rewards[i]
                probs = policy_model(state)
                log_prob = tf.math.log(probs[0][action])
                policy_loss += -log_prob * reward

        grads = tape.gradient(policy_loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))
        episode_score = np.sum(rewards)
        total_reward = episode_score
        episodic_improvements.append(total_reward)
        print("Episode {}: Score = {}".format(episode, episode_score))
        avg_reward = np.mean(episodic_improvements)
        with summary_writer.as_default():
            tf.summary.scalar('epoch_loss', policy_loss,
                              step=episode)
            tf.summary.scalar('1/epoch_total_reward', total_reward,
                              step=episode)
            tf.summary.scalar('1/epoch_average_reward', avg_reward,
                              step=episode)
        if total_reward > avg_reward:
            policy_model.save_weights(
                'weights/'+env.unwrapped.spec.id+'reinforce_model.h5')
            print('...model save success...')
