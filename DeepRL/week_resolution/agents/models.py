import tensorflow as tf
import tensorflow_probability as tfp
from agents.params import *

tf.random.set_seed(SEED)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model():
    inputs = tf.keras.layers.Input(in_dim)
    x = tf.keras.layers.Conv2D(
        16, kernel_size=8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(out_dim)(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    m = model()
    # m.output.activation = 'softmax'
    m.summary()
