import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import sys
import tqdm
import gc

from agents.params import *
from agents.models import *
from agents.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(SEED)


class DQNAgent():
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
        self.loss = tf.keras.losses.Huber()

    @tf.function
    def greedy_act(self, state):
        logits = self.q_network(state, training=False)
        action = tf.math.argmax(logits)
        return action[0]

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return self.greedy_act(state)

    def update_weights(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        indices = tf.range(BATCH_SIZE, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            q_pred = tf.gather_nd(self.q_network(states),
                                  indices=action_indices)
            q_next = self.target_q_network(next_states)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                GAMMA*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - dones)

            loss = self.loss(q_target, q_pred)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables))
        return loss

    def train(self):
        minibatch = []
        for ind in np.random.choice(len(self.memory), BATCH_SIZE, replace=False):
            minibatch.append(self.memory[ind])
        states = np.array([experience[0]
                          for experience in minibatch], np.uint8)
        actions = np.array([experience[1]
                           for experience in minibatch], np.int32)
        rewards = np.array([experience[2]
                           for experience in minibatch], np.float32)
        next_states = np.array([experience[3]
                               for experience in minibatch], np.uint8)
        dones = np.array([experience[4]
                         for experience in minibatch], np.float32)

        loss = self.train_step(states, actions, rewards, next_states, dones)
        return loss


def DQN(env):
    player = DQNAgent(env)
    log_dir = os.path.join(
        "logs/logs_DQN", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    episodic_improvements = []
    counter = 0
    heat = False
    flush_print = 200
    player.update_weights()
    loss = 0
    for episode in tqdm.tqdm(range(1, NUM_EPISODES+1)):
        state = player.env.reset()[0]
        done = False
        truncated = False
        total_reward = 0
        frames = 0
        while not (done or truncated):
            if counter % TARGET_UPDATE_FREQUENCY == 0:
                player.update_weights()

            counter += 1
            action = player.act(tf.expand_dims(np.array(state).T, axis=0))
            next_state, reward, done, truncated, info = player.env.step(action)
            player.memory.append(
                (np.array(state).T, action, reward, np.array(next_state).T, done))

            if len(player.memory) > MEMORY_CAPACITY:
                player.memory.pop(0)
            state = next_state
            total_reward += reward
            frames += 1

            if counter % UPDATE_FREQUENCY == 0 and len(player.memory) >= MIN_REPLAY_BUFFER_SIZE:
                loss = player.train()
            if frames >= MAX_STEPS:
                break

        if len(player.memory) >= MIN_REPLAY_BUFFER_SIZE:
            if not heat:
                print('Heating Finished!!!!')
                heat = True
                flush_print = 50
            player.epsilon = max(EPSILON_END, player.epsilon*EPSILON_DECAY)

        episodic_improvements.append(total_reward)
        avg_reward = np.mean(episodic_improvements[-WINDOW_SIZE:])
        # print(
        #     f"\rEpisode {episode + 1}: Total reward = {total_reward} Epsilon:{player.epsilon}", end=' ')
        # sys.stdout.flush()
        if episode % flush_print == 0:
            print(
                f"Episode {episode + 1}: Total reward = {total_reward} Epsilon:{player.epsilon}")
        with summary_writer.as_default():
            tf.summary.scalar('epoch_loss', loss,
                              step=episode)
            tf.summary.scalar('1/epoch_total_reward', total_reward,
                              step=episode)
            tf.summary.scalar('1/epoch_average_reward', avg_reward,
                              step=episode)
        if total_reward > avg_reward:
            player.target_q_network.save_weights(
                'weights/'+env.unwrapped.spec.id+'DQN_model.h5')
        gc.collect()
    print()
