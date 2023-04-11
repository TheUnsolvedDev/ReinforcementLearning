import tensorflow as tf
import tensorflow_probability as tfp
from params import *


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model(in_dim=100, out_dim=10, hidden_units=[32, 32], advantage=False):
    inputs = tf.keras.layers.Input(in_dim)
    x = tf.keras.layers.BatchNormalization()(inputs)
    for units in hidden_units:
        x = tf.keras.layers.Dense(
            units, activation='tanh', kernel_initializer='RandomNormal')(x)
        x = tf.keras.layers.Dropout(0.15)(x)

    if not advantage:
        outputs = tf.keras.layers.Dense(
            out_dim, kernel_initializer='RandomNormal')(x)
    else:
        q_values = tf.keras.layers.Dense(
            out_dim, kernel_initializer='RandomNormal')(x)
        values = tf.keras.layers.Dense(1, kernel_initializer='RandomNormal')(x)
        outputs = [values, q_values]
    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    m = model(advantage=True)
    # m.output.activation = 'softmax'
    m.summary()
