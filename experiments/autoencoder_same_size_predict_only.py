import sys
import tensorflow as tf
import ai
from ai.utils import display_compare_results_pyplot2, load_model

if __name__ == '__main__':
    # Without this computation on the GPU may not work.
    # This has to be executed before any other tensorflow function as it reconfigures the device.
    ai.utils.allow_memory_growth()

    _, dataset_test = ai.datasets.landscapes.load_dataset(image_size=(256, 256))

    if len(sys.argv) > 1:
        model = load_model(name=sys.argv[1])
    else:
        model = load_model()
    for batch in dataset_test:
        inputs = tf.expand_dims(batch[:, :, :, 0], axis=-1)
        pred = model.predict(inputs)
        display_compare_results_pyplot2(inputs, batch, pred, 3)
        break
