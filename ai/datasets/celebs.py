import os
import PIL
import PIL.Image
import tensorflow as tf

from ..utils import prepare_image_as_input2


# This dataset is quite large so we can limit element count to speed up development
def load_dataset(train_examples_count=10, validation_examples_count=10, batch_size=5):
    dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
        r'C:\Users\pjtom\Downloads\img_align_celeba',
        validation_split=0.2,
        subset='training',
        seed=123,
        batch_size=batch_size
    )

    dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
        r'C:\Users\pjtom\Downloads\img_align_celeba',
        validation_split=0.2,
        subset='validation',
        seed=123,
    )

    # if train_examples_count < 0 or train_examples_count > info.splits['train'].num_examples:
    #     train_examples_count = info.splits['train'].num_examples
    #
    # if validation_examples_count < 0 or validation_examples_count > info.splits['validation'].num_examples:
    #     validation_examples_count = info.splits['validation'].num_examples

    print('')
    dataset_train = dataset_train\
        .take(train_examples_count)\
        .shuffle(train_examples_count)\
        .map(prepare_image_as_input2)\
        # .batch(batch_size)

    dataset_validation = dataset_validation\
        .take(validation_examples_count)\
        .shuffle(validation_examples_count) \
        .map(prepare_image_as_input2)\
        # .batch(batch_size)

    # Zostawiam tymczasowo, żeby było łatwo debuggerem podejrzeć jak wyglądają elementy w batchu
    # from ..utils import display_ycbcr_batch_pyplot, display_bw_batch_pyplot
    # for x in tfds.as_numpy(dataset_train):
    #     display_bw_batch_pyplot(x['tensor_bw'][:5])
    #     display_ycbcr_batch_pyplot(x['tensor_org'][:5])
    #     pass

    # for x in dataset_train:
    #     # display_bw_batch_pyplot(x['tensor_bw'])
    #     display_ycbcr_batch_pyplot(x['tensor_org'])
    #     pass

    return dataset_train, dataset_validation
