import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import gc
import sys
import tqdm

from agents.params import *
from agents.models import *
from agents.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DoubleDQNAgent():
    def __init__(self, env):
        self.env = env
        self.num_actions = out_dim

        self.q_network = model()
        self.q_network.summary()
        self.target_q_network = model()
        self.target_q_network.summary()

        self.optimizer = tf.keras.optimizers.RMSprop(ALPHA)
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.memory = []
        self.epsilon = EPSILON_START

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.q_network(
            tf.convert_to_tensor(state, dtype=tf.float32))
        return np.argmax(q_values)

    def update_weights(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return -100

        minibatch = []
        for ind in np.random.choice(len(self.memory), BATCH_SIZE, replace=False):
            minibatch.append(self.memory[ind])
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        indices = tf.range(BATCH_SIZE, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            q_pred = tf.gather_nd(self.q_network(states),
                                  indices=action_indices)
            q_next = self.target_q_network(next_states)
            q_eval = self.q_network(next_states)

            max_actions = tf.math.argmax(q_eval, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                GAMMA*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - dones)

            loss = tf.keras.losses.MSE(q_pred, q_target)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables))

        return loss


def DoubleDQN(env):
    player = DoubleDQNAgent(env)
    log_dir = os.path.join(
        "logs/logs_DoubleDQN", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    episodic_improvements = []
    counter = 0

    player.update_weights()
    for episode in tqdm.tqdm(range(1, NUM_EPISODES+1)):
        state = player.env.reset()[0]
        states = join_frames(state, initial=True)
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            if counter % TARGET_UPDATE_FREQUENCY == 0:
                player.update_weights()
            counter += 1
            action = player.act([states])
            next_state, reward, done, truncated, info = player.env.step(action)
            next_states = join_frames(next_state, states)
            player.memory.append((states, action, reward, next_states, done))
            if len(player.memory) > MEMORY_CAPACITY:
                player.memory.pop(0)
            state = next_state
            states = next_states
            total_reward += reward
            loss = player.train()
        player.epsilon = max(EPSILON_END, player.epsilon-EPSILON_DECAY)

        episodic_improvements.append(total_reward)
        avg_reward = np.mean(episodic_improvements[-WINDOW_SIZE:])
        print(
            f"\rEpisode {episode + 1}: Total reward = {total_reward} Epsilon:{player.epsilon}", end=' ')
        sys.stdout.flush()
        if episode % 100 == 0:
            print(
                f"Episode {episode + 1}: Total reward = {total_reward} Epsilon:{player.epsilon}")
        with summary_writer.as_default():
            tf.summary.scalar('epoch_loss', loss,
                              step=episode)
            tf.summary.scalar('1/epoch_total_reward', total_reward,
                              step=episode)
            tf.summary.scalar('1/epoch_average_reward', avg_reward,
                              step=episode)
        if total_reward >= avg_reward:
            player.target_q_network.save_weights(
                'weights/'+env.unwrapped.spec.id+'DoubleDQN_model.h5')
        gc.collect()
    print()
