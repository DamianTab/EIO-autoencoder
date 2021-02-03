import tensorflow as tf
import ai
import matplotlib.pyplot as plt
import numpy as np

@tf.function
def query(images, classes, training):
    pass


@tf.function
def train(images, classes):
    pass


if __name__ == '__main__':
    # Without this computation on the GPU may not work.
    # This has to be executed before any other tensorflow function as it reconfigures the device.
    ai.utils.allow_memory_growth()

    # Elements in those datasets represents batches.
    # Each batch is a dictionary that has two keys:
    # tensor_org - is the original colorful image in YCbCr color space; expected network output
    # tensor_bw - is the black and white representation; network input
    # load dataset
    dataset_train, dataset_test = ai.datasets.load_dataset()


    # # train
    # autoencoder = ai.models.AutoEncoder()
    # autoencoder.compile(optimizer='adam',
    #                     loss='mean_squared_error',
    #                     metrics=['accuracy'])
    # autoencoder.fit(dataset_train, x_train, batch_size=64, epochs=2)
    #
    #
    # # result
    # x_plot = x_test.astype('float32') / 255.
    # x_plot = np.reshape(x_plot, (len(x_test), 28, 28))
    # x_predicted = autoencoder.predict(x_test)
    # x_predicted = x_predicted.astype('float32') / 255.
    # x_predicted = np.reshape(x_predicted, newshape=(x_predicted.shape[0], 28, 28))
    #
    #
    # fig, ax = plt.subplots(nrows=10, ncols=2)
    # for i, row in enumerate(ax):
    #     row[0].imshow(x_plot[i, :, :], cmap='gray')
    #     row[1].imshow(x_predicted[i, :, :], cmap='gray')
    # plt.show()
    #
