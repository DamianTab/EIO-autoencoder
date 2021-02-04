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

    dataset_train, dataset_test = ai.datasets.load_dataset(train_examples_count=10000, validation_examples_count=20,
                                                           batch_size=1000)

    model = ai.models.AutoEncoder()
    optimizer = tf.optimizers.Adam(0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    epoch_count = 100
    for i in range(epoch_count):
        print(f'=== Epoch {i} ===')
        for batch in dataset_train:
            model.fit(batch['tensor_bw'], batch['tensor_org'])
        for batch in dataset_test:
            results = model.evaluate(batch['tensor_bw'], batch['tensor_org'])
            print(f'Validation loss: {results[0]}')
            print(f'Validation MAE: {results[1]}')
            break

    for batch in dataset_test:
        pred = model.predict(batch['tensor_bw'])
        display_compare_results_pyplot(batch['tensor_org'], pred, 10)
        break

    save_model(model)
