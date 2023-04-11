import tensorflow as tf
import tensorflow_probability as tfp
from agents.params import *

tf.random.set_seed(SEED)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model(in_dim=100, out_dim=10, hidden_units=[128, 128], advantage=False):
    inputs = tf.keras.layers.Input(in_dim)
    if hidden_units:
        x = tf.keras.layers.Dense(
            hidden_units[0], activation='relu', kernel_initializer='he_uniform')(inputs)
        for units in hidden_units[1:]:
            x = tf.keras.layers.Dense(
                units, activation='tanh', kernel_initializer='he_uniform')(x)
    if not advantage:
        outputs = tf.keras.layers.Dense(
            out_dim, kernel_initializer='he_uniform', kernel_regularizer='l2', bias_regularizer='l2')(x)
    else:
        q_values = tf.keras.layers.Dense(
            out_dim, kernel_initializer='he_uniform', kernel_regularizer='l2', bias_regularizer='l2')(x)
        values = tf.keras.layers.Dense(
            1, kernel_initializer='he_uniform', kernel_regularizer='l2', bias_regularizer='l2')(x)
        outputs = [values, q_values]
    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    m = model(advantage=True)
    # m.output.activation = 'softmax'
    m.summary()
