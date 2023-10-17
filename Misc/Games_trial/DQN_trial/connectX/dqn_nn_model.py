import tensorflow as tf
import numpy as np

from params import *


class DeepModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DeepModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='he_normal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='tanh', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Nadam(lr)
        self.gamma = gamma
        self.model = DeepModel(num_states, hidden_units, num_actions)
        self.experience = {'inputs': [], 'a': [], 'r': [],
                           'inputs2': [], 'done': []}  # The buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    # @tf.function
    def train(self, TargetNet):
        if len(self.experience['inputs']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(
            self.experience['inputs']), size=self.batch_size)
        states = np.asarray([self.experience['inputs'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['inputs2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        Move_Validity = states_next[:, :self.num_actions] == 0
        Next_Q_Values = TargetNet.predict(states_next)
        Next_Q_Values = np.where(Move_Validity, Next_Q_Values, Discard_Q_Value)
        value_next = -np.max(Next_Q_Values, axis=1)

        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(
                states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(
                tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, state, epsilon):
        prediction = self.predict(np.atleast_2d(
            self.preprocess(state)))[0].numpy()
        if np.random.random() < epsilon:
            return int(np.random.choice([c for c in range(self.num_actions) if state['board'][c] == 0])), prediction
        else:
            for i in range(self.num_actions):
                if state['board'][i] != 0:
                    prediction[i] = Discard_Q_Value
            return int(np.argmax(prediction)), prediction

    def add_experience(self, exp):
        if len(self.experience['inputs']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        ref_model = tf.keras.Sequential()

        ref_model.add(self.model.input_layer)
        for layer in self.model.hidden_layers:
            ref_model.add(layer)
        ref_model.add(self.model.output_layer)

        ref_model.load_weights(path)

    def preprocess(self, state):
        print(state.keys())
        return np.array([1 if val == state.mark else 0 if val == 0 else -1 for val in state['board']])


if __name__ == "__main__":
    agent = DQN()
