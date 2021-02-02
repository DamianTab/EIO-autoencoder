import tensorflow as tf
import ai

@tf.function
def query(images, classes, training):
    pass


@tf.function
def train(images, classes):
    pass


if __name__ == '__main__':
    autoencoder = ai.models.AutoEncoder()
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['accuracy'])
    autoencoder.fit(x_train, x_train, batch_size=128, epochs=3)
    pass
