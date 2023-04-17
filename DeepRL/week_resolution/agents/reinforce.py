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


@tf.function
def act(actor, state):
    prob = actor(tf.expand_dims(state, axis=0))
    logits = tf.math.log(prob)
    action = tf.random.categorical(logits, 1)
    return int(action[0, 0])


def reinforce(env):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=ALPHA)
    policy_model = model()
    policy_model.summary()
    policy_model.layers[-1].activation = tf.keras.activations.softmax
    log_dir = os.path.join(
        "logs/logs_reinforce", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    episodic_improvements = []

    for episode in tqdm.tqdm(range(1, NUM_EPISODES+1)):
        all_states, all_actions, all_rewards = [], [], []
        for i in range(NUM_TRAJECTORIES):
            states, actions, rewards = [], [], []
            state = env.reset()[0]
            done = False
            truncated = False
            while not (done or truncated):
                states.append(np.array(state).T)
                action = act(policy_model, np.array(state).T)
                actions.append(action)
                next_state, reward, done, truncated, info = env.step(
                    action)
                rewards.append(reward)
                state = next_state

            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(discounted_rewards(rewards))
            episode_score = np.sum(rewards)

            del states
            del actions
            del rewards
            gc.collect()

        with tf.GradientTape() as tape:
            policy_loss = 0
            for batch_start in range(0, len(all_states), BATCH_SIZE):
                batch_end = batch_start + BATCH_SIZE
                batch_states = np.array(all_states[batch_start:batch_end])
                batch_actions = np.array(all_actions[batch_start:batch_end])
                batch_rewards = np.array(all_rewards[batch_start:batch_end])

                batch_log_probs = tf.reduce_sum(tf.one_hot(
                    batch_actions, depth=policy_model.output_shape[-1]) * tf.math.log(policy_model(batch_states)), axis=-1)
                batch_policy_loss = - \
                    tf.reduce_sum(batch_log_probs * batch_rewards)
                policy_loss += tf.reduce_sum(batch_policy_loss)
        gc.collect()

        del all_states
        del all_actions
        del all_rewards
        gc.collect()

        grads = tape.gradient(policy_loss, policy_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

        total_reward = episode_score
        episodic_improvements.append(total_reward)
        avg_reward = np.mean(episodic_improvements[-WINDOW_SIZE:])
        print(
            f"\rEpisode {episode + 1}: Total reward = {total_reward}", end=' ')
        sys.stdout.flush()
        if episode % 100 == 0:
            print(
                f"Episode {episode + 1}: Total reward = {total_reward}")
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
    print()
