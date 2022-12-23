import numpy as np
import tensorflow as tf

from params import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def lenet5(input_shape=(*IMG_SIZE, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Lambda(
        lambda i: i/255.0, input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=6, strides=3, kernel_size=(
        5, 5), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=3, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='relu'))
    model.add(tf.keras.layers.Dense(units=84, activation='relu'))
    model.add(tf.keras.layers.Dense(
        units=env.action_space.n, activation='softmax'))
    return model


if __name__ == '__main__':
    model = lenet5()
    model.summary()
    tf.keras.utils.plot_model(model, to_file=lenet5.__name__+'.png')
