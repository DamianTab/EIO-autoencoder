import tensorflow as tf

def mse_loss(outputs, features):
    return tf.reduce_mean(tf.square(outputs - features))
