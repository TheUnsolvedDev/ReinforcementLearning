import tensorflow as tf

from params import *


def cnn(input_shape, output_shape):
    inputs = tf.keras.layers.Input(input_shape)
    conv1 = tf.keras.layers.Conv2D(
        32, kernel_size=8, strides=4, activation='relu')(inputs)
    conv2 = tf.keras.layers.Conv2D(
        64, kernel_size=4, strides=2, activation='relu')(conv1)
    conv3 = tf.keras.layers.Conv2D(
        64, kernel_size=3, strides=1, activation='relu')(conv2)
    flatten = tf.keras.layers.Flatten()(conv3)
    dense1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
    outputs = tf.keras.layers.Dense(output_shape)(dense1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def dnn(input_shape, output_shape):
    inputs = tf.keras.layers.Input(input_shape)
    dense1 = tf.keras.layers.Dense(512, activation='relu')(inputs)
    dense2 = tf.keras.layers.Dense(512, activation='relu')(dense1)
    outputs = tf.keras.layers.Dense(output_shape)(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = dnn(INPUT_SHAPE, ACTION_SIZE)
    model.summary()
