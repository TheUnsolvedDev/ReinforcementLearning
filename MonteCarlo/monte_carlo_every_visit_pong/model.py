import tensorflow as tf
from params import *


def cnn_model(image_size=IMAGE_SIZE):
    inputs = tf.keras.layers.Input(image_size)
    x = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
    x = tf.keras.layers.Conv2D(16, kernel_size=(
        8, 8), strides=4, activation='relu', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(
        4, 4), strides=2, activation='relu', use_bias=False)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu', use_bias=False)(x)
    outputs = tf.keras.layers.Dense(6, activation='linear', use_bias=False)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    mod = cnn_model()
    mod.summary()
    tf.keras.utils.plot_model(mod, to_file='model.png', show_shapes=True)
