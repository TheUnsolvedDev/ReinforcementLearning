import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from agents.params import *
from agents.models import *
from agents.utils import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class agent():
    def __init__(self):
        self.decay = tf.optimizers.schedules.ExponentialDecay(
            ALPHA, NUM_EPISODES, 0.999)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.decay)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=self.decay)
        self.actor = model(in_dim, out_dim)
        self.actor.layers[-1].activation = tf.keras.activations.softmax
        self.critic = model(in_dim, 1)
        self.log_prob = None

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        with tf.GradientTape(persistent=True) as tape1, tf.GradientTape(persistent=True) as tape2:
            p = self.actor(state, training=True)
            v = self.critic(state, training=True)
            vn = self.critic(next_state, training=True)
            td = reward + GAMMA*vn*(1-int(done)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td**2
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(
            zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


def NaiveActorCritic(env):
    log_dir = os.path.join(
        "logs_ActorCritic", env.unwrapped.spec.id+'_events')
    summary_writer = tf.summary.create_file_writer(logdir=log_dir)
    episodic_improvements = []
    agentoo7 = agent()

    for episode in range(NUM_EPISODES):
        state = env.reset()[0]
        episode_reward = 0
        while True:
            action = agentoo7.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            aloss, closs = agentoo7.learn(
                state, action, reward, next_state, done)
            if done or truncated:
                break
            loss_value = aloss + closs
            episode_reward += reward
        total_reward = episode_reward
        episodic_improvements.append(episode_reward)
        avg_reward = np.mean(episodic_improvements)
        print(
            f"Episode {episode + 1}: Total reward = {total_reward} loss: {loss_value[0][0]}")
        with summary_writer.as_default():
            tf.summary.scalar('epoch_loss', loss_value[0][0],
                              step=episode)
            tf.summary.scalar('1/epoch_total_reward', total_reward,
                              step=episode)
            tf.summary.scalar('1/epoch_average_reward', avg_reward,
                              step=episode)
        if total_reward >= avg_reward:
            agentoo7.actor.save_weights(
                'weights/'+env.unwrapped.spec.id+'ActorCritic_model.h5')
            print('...model save success...')
