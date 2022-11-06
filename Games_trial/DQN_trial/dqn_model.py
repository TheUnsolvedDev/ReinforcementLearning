
import tensorflow as tf
import numpy as np
import random

BATCH_SIZE = 128


class ReplayBuffer:
    def __init__(self) -> None:
        self.experience = []

    def store_gameplay(self, state, action, reward, done, next_state):
        self.experience.append((state, action, reward, done, next_state))

    def sample_batch(self):
        batch_size = min(BATCH_SIZE, len(self.experience))
        sampled_games_batch = random.sample(self.experience, batch_size)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []

        for games in sampled_games_batch:
            states.append(games[0])
            actions.append(games[1])
            rewards.append(games[2])
            dones.append(games[3])
            next_states.append(games[4])

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


def linear_model(input_shape=(4), output_shape=2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    hidden = tf.keras.layers.Dense(32, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(output_shape, activation='linear')(hidden)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model


class QN:
    def __init__(self, decay=0, epsilon=1, gamma=0.95):
        self.Q = linear_model()
        self.target_Q = linear_model()
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay = decay

    def policy(self, state, train=True):
        if train:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, 2)
            state = tf.expand_dims(state, axis=0)
            action = self.Q(state)
            action = np.argmax(action.numpy()[0], axis=0)
        else:
            state = tf.expand_dims(state, axis=0)
            action = self.Q(state)
            action = np.argmax(action.numpy()[0], axis=0)
        return action

    def inference(self):
        self.Q = tf.keras.models.load_model('Q.h5')

    def train(self, batch):
        state_batch, action_batch, reward_batch, done_batch, next_state_batch = batch
        current_q = self.Q(state_batch).numpy()
        target_q = np.copy(current_q)
        next_q = self.target_Q(next_state_batch).numpy()

        max_next_q = np.amax(next_q, axis=1)
        target_q = tf.add(
            reward_batch, (1-np.array(done_batch, dtype=np.int8))*self.gamma*max_next_q)

        training_history = self.Q.fit(x=state_batch, y=target_q, verbose=0)
        loss = training_history.history['loss']
        return loss


if __name__ == '__main__':
    model = linear_model()
    model.summary()
