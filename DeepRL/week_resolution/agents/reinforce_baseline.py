import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import sys
import tqdm
import gc

from memory_profiler import profile
from agents.params import *
from agents.models import *
from agents.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(SEED)


@tf.function
def act(actor, state):
    prob = actor(tf.expand_dims(state, axis=0), training=False)
    logits = tf.math.log(prob)
    action = tf.random.categorical(logits, 1)
    return int(action[0, 0])


def reinforce_baseline(env):
    optimizer = tf.keras.optimizers.Adam(ALPHA)
    log_dir = os.path.join(
        "logs/logs_reinforce_baseline", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    policy_model = model(out_dim)
    policy_model.layers[-1].activation = tf.keras.activations.softmax
    policy_model.summary()

    value_model = model(out=1)
    value_model.summary()
    episodic_improvements = []

    for episode in tqdm.tqdm(range(NUM_EPISODES)):
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
                states.append(np.array(state).T)
                action = act(policy_model, np.array(state).T)
                next_state, reward, done, truncated, info = env.step(
                    action)
                actions.append(action)
                rewards.append(reward)
                state = next_state

            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            episode_score = np.sum(rewards)

            del states
            del actions
            del rewards
            gc.collect()

        states = np.concatenate(states_list)
        actions = np.concatenate(actions_list)
        rewards = np.concatenate(rewards_list)
        discounted = discounted_rewards(rewards, GAMMA)
        baselines = calculate_baselines(rewards)
        total_rewards = sum(rewards)
        episodic_improvements.append(total_rewards)

        with tf.GradientTape() as tape:
            policy_loss = 0
            value_loss = 0

            for batch_start in range(0, len(states), BATCH_SIZE):
                batch_end = batch_start + BATCH_SIZE
                states = np.array(states[batch_start:batch_end])
                actions = np.array(actions[batch_start:batch_end])
                discounted = np.array(discounted[batch_start:batch_end])
                baselines = np.array(baselines[batch_start:batch_end])

                # Policy loss
                probs = policy_model(states)
                log_probs = tf.math.log(tf.gather_nd(probs, tf.stack(
                    (tf.range(tf.shape(actions)[0]), actions), axis=1)))
                advantages = discounted - baselines
                policy_loss += -tf.reduce_mean(log_probs * advantages)

                # Value loss
                values = value_model(states)
                advantages = discounted - values[:, 0]
                value_loss += tf.reduce_mean(tf.square(advantages))

            # Combined loss
            loss = policy_loss + 0.5 * value_loss
        gc.collect()

        del states_list
        del actions_list
        del rewards_list

        del states
        del actions
        del rewards
        del discounted
        del baselines
        gc.collect()

        gradients = tape.gradient(
            loss, policy_model.trainable_variables + value_model.trainable_variables)
        optimizer.apply_gradients(zip(
            gradients, policy_model.trainable_variables + value_model.trainable_variables))
        total_reward = episode_score
        episodic_improvements.append(total_reward)
        avg_reward = np.mean(episodic_improvements[-WINDOW_SIZE:])
        # print(
        #     f"\rEpisode {episode + 1}: Total reward = {total_reward}", end=' ')
        # sys.stdout.flush()
        if episode % 100 == 0:
            print(
                f"Episode {episode + 1}: Total reward = {total_reward}")
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
    print()
