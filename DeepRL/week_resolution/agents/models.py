import tensorflow as tf
import tensorflow_probability as tfp
from agents.params import *

tf.random.set_seed(SEED)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model(out=out_dim):
    inputs = tf.keras.layers.Input(in_dim)
    x = tf.keras.layers.Rescaling(1/255.0)(inputs)
    x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.), activation='relu', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.), activation='relu', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.), activation='relu', use_bias=False)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(out)(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    m = model()
    # m.output.activation = 'softmax'
    m.summary()
