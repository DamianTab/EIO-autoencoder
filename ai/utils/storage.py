import os
import tensorflow as tf
from pathlib import Path


def save_model(model, directory=None, name='Auto_Encoder'):
    if directory is None:
        directory = os.path.join(Path(os.getcwd()).parent, 'saved_models')
    if not os.path.exists(directory):
        os.mkdir(directory)
    model.save(os.path.join(directory, name))


def load_model(directory=None, name='Auto_Encoder'):
    if directory is None:
        directory = os.path.join(Path(os.getcwd()).parent, 'saved_models')
    return tf.keras.models.load_model(os.path.join(directory, name), compile=False)
