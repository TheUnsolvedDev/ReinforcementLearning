import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from agents.params import *
from agents.models import *
from agents.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def act(actor, state):
    prob = actor(np.array([state]))
    prob = prob.numpy()
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    action = dist.sample()
    return int(action.numpy()[0])


def reinforce_baseline(env):
    optimizer = tf.keras.optimizers.Adam(ALPHA)
    log_dir = os.path.join(
        "logs_reinforce_baseline", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    policy_model = model(in_dim, out_dim)
    policy_model.layers[-1].activation = tf.keras.activations.softmax
    policy_model.summary()

    value_model = model(in_dim, 1)
    value_model.summary()
    episodic_improvements = []
    for episode in range(NUM_EPISODES):
        states_list = []
        actions_list = []
        rewards_list = []

        for i in range(NUM_TRAJECTORIES):
            states = []
            actions = []
            rewards = []
            state = env.reset()[0]
            done = False
            truncated = False
            while not (done or truncated):
                states.append(state)
                action = act(policy_model, state)
                state, reward, done, truncated, info = env.step(
                    action)
                actions.append(action)
                rewards.append(reward)

            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)

        states = np.concatenate(states_list)
        actions = np.concatenate(actions_list)
        rewards = np.concatenate(rewards_list)
        discounted = discounted_rewards(rewards, GAMMA)
        baselines = calculate_baselines(rewards)
        total_rewards = sum(rewards)
        episodic_improvements.append(total_rewards)

        with tf.GradientTape() as tape:
            policy_loss = 0
            for i in range(len(states)):
                state = np.array([states[i]])
                action = actions[i]
                reward = discounted[i]
                baseline = baselines[i]
                probs = policy_model(state)
                log_prob = tf.math.log(probs[0][action])
                advantage = reward - baseline
                policy_loss += -log_prob * advantage

            value_loss = 0
            for i in range(len(states)):
                state = np.array([states[i]])
                reward = discounted[i]
                value = value_model(state)[0][0]
                advantage = reward - value
                value_loss += tf.square(advantage)
            loss = policy_loss + 0.5 * value_loss

        gradients = tape.gradient(
            loss, policy_model.trainable_variables + value_model.trainable_variables)
        optimizer.apply_gradients(zip(
            gradients, policy_model.trainable_variables + value_model.trainable_variables))
        total_reward = sum(rewards)
        print(f"Episode {episode}: Total reward = {total_reward}")
        avg_reward = np.mean(episodic_improvements)
        with summary_writer.as_default():
            tf.summary.scalar('epoch_loss', loss,
                              step=episode)
            tf.summary.scalar('1/epoch_total_reward', total_reward,
                              step=episode)
            tf.summary.scalar('1/epoch_average_reward', avg_reward,
                              step=episode)
        if total_reward > avg_reward:
            policy_model.save_weights(
                'weights/'+env.unwrapped.spec.id+'reinforce_baseline_model.h5')
            print('...model save success...')
