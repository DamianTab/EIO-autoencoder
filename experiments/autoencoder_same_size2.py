import sys

import tensorflow as tf

import ai
from ai.utils import display_compare_results_pyplot, load_model, save_model

if __name__ == '__main__':
    # Without this computation on the GPU may not work.
    # This has to be executed before any other tensorflow function as it reconfigures the device.
    ai.utils.allow_memory_growth()

    # Elements in those datasets represents batches.
    # Each batch is a dictionary that has two keys:
    # tensor_org - is the original colorful image in YCbCr color space; expected network outputs
    # tensor_bw - is the black and white representation; network input
    # load dataset
    dataset_train, dataset_test = ai.datasets.load_dataset(train_examples_count=1024 * 8, validation_examples_count=128)

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        model = load_model(name=model_name)
    else:
        model_name = 'Test'
        model = ai.models.AutoEncoder(batch_normalization=True, dropout_rate=0.1, l2_regularization=0.1)

    optimizer = tf.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])  # MeanAbsoluteError

    best_metric = 1
    epoch_count = 100
    for i in range(epoch_count):
        print(f'=== Epoch {i} ===')
        for batch in dataset_train:
            inputs = tf.expand_dims(batch['tensor_org'][:, :, :, 0], axis=-1)
            model.fit(inputs, batch['tensor_org'][:, :, :, 1:], batch_size=64)
        for batch in dataset_test:
            inputs = tf.expand_dims(batch['tensor_org'][:, :, :, 0], axis=-1)
            results = model.evaluate(inputs, batch['tensor_org'][:, :, :, 1:], batch_size=64)

        if best_metric > results[1]:
            best_metric = results[1]
            save_model(model, name="Best")

        save_model(model, name=model_name)
        print(f'Validation loss: {results[0]}')
        print(f'Validation MAE: {results[1]}\n')

    for batch in dataset_test:
        inputs = tf.expand_dims(batch['tensor_org'][:, :, :, 0], axis=-1)
        pred = model.predict(inputs)
        display_compare_results_pyplot(batch['tensor_org'], pred, 10)
        break
