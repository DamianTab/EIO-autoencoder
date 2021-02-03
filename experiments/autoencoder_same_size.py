import time

import tensorflow as tf

import ai
from ai.utils import display_ycbcr_batch_pyplot


# @tf.function
def query(model, inputs, features):
    outputs = model(inputs)
    loss = ai.losses.mse_loss(outputs, features)
    return outputs, loss


# @tf.function
def train(optimizer, model, inputs, features):
    with tf.GradientTape() as tape:
        outputs, loss = query(model, inputs, features)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return outputs, loss


if __name__ == '__main__':
    # Without this computation on the GPU may not work.
    # This has to be executed before any other tensorflow function as it reconfigures the device.
    ai.utils.allow_memory_growth()

    # Elements in those datasets represents batches.
    # Each batch is a dictionary that has two keys:
    # tensor_org - is the original colorful image in YCbCr color space; expected network outputs
    # tensor_bw - is the black and white representation; network input
    # load dataset

    # dataset_train, dataset_test = ai.datasets.load_dataset(train_examples_count=1024, validation_examples_count=128,
    #                                                        batch_size=64)
    dataset_train, dataset_test = ai.datasets.load_dataset(train_examples_count=1024, validation_examples_count=10,
                                                           batch_size=2)

    model = ai.models.AutoEncoder()
    optimizer = tf.optimizers.Adam(0.001)
    epoch_count = 2

    for i in range(epoch_count):
        start = time.time()
        for batch in dataset_train:
            inputs = batch['tensor_bw']
            features = batch['tensor_org']
            outputs, loss = train(optimizer, model, inputs, features)

        display_ycbcr_batch_pyplot(features)
        display_ycbcr_batch_pyplot(outputs)
        end = time.time()
        print(loss)
        print(f'time: {(end - start)}')

