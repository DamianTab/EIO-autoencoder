import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist


def load_dataset(split='train[:5%]', batch_size=64):
    # x_train = tfds.load(name="mnist", split=tfds.Split.TRAIN) \
    #     .shuffle(1024) \
    #     .batch(batch_size) \
    #     .prefetch(tf.data.experimental.AUTOTUNE)
    #
    # x_test = tfds.load(name="mnist", split=tfds.Split.TEST) \
    #     .shuffle(1024) \
    #     .batch(batch_size) \
    #     .prefetch(tf.data.experimental.AUTOTUNE)
    #
    # return x_train, x_test

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:, :, :, np.newaxis].astype('float32')
    x_test = x_test[:, :, :, np.newaxis].astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test
