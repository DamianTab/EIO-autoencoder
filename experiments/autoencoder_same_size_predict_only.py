import sys

import ai
from ai.utils import display_compare_results_pyplot2, load_model

if __name__ == '__main__':
    # Without this computation on the GPU may not work.
    # This has to be executed before any other tensorflow function as it reconfigures the device.
    ai.utils.allow_memory_growth()

    # Elements in those datasets represents batches.
    # Each batch is a dictionary that has two keys:
    # tensor_org - is the original colorful image in YCbCr color space; expected network outputs
    # tensor_bw - is the black and white representation; network input
    # load dataset

    _, dataset_test = ai.datasets.landscapes.load_dataset(image_size=(1024, 1800))

    if len(sys.argv) > 1:
        model = load_model(name=sys.argv[1])
    else:
        model = load_model()
    for batch in dataset_test:
        pred = model.predict(batch['tensor_bw'])
        display_compare_results_pyplot2(batch['tensor_bw'], batch['tensor_org'], pred, 1)
        break
