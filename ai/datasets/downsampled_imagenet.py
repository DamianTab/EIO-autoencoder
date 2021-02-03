import tensorflow_datasets as tfds

from ..utils import prepare_image_as_input


# This dataset is quite large so we can limit element count to speed up development
def load_dataset(train_examples_count=10, validation_examples_count=10, batch_size=5):
    # datasets, info = tfds.load('downsampled_imagenet/64x64', with_info=True)
    datasets, info = tfds.load('imagenet_resized/8x8', with_info=True)
    dataset_train = datasets['train']
    dataset_validation = datasets['validation']

    if train_examples_count < 0 or train_examples_count > info.splits['train'].num_examples:
        train_examples_count = info.splits['train'].num_examples

    if validation_examples_count < 0 or validation_examples_count > info.splits['validation'].num_examples:
        validation_examples_count = info.splits['validation'].num_examples

    dataset_train = dataset_train\
        .take(train_examples_count)\
        .shuffle(train_examples_count)\
        .map(prepare_image_as_input)\
        .batch(batch_size)

    dataset_validation = dataset_validation\
        .take(validation_examples_count)\
        .shuffle(validation_examples_count)\
        .map(prepare_image_as_input)\
        .batch(batch_size)

    # Zostawiam tymczasowo, żeby było łatwo debuggerem podejrzeć jak wyglądają elementy w batchu
    from ..utils import display_ycbcr_batch_pyplot, display_bw_batch_pyplot
    # for x in tfds.as_numpy(dataset_train):
    #     # display_bw_batch_pyplot(x['tensor_bw'])
    #     # display_ycbcr_batch_pyplot(x['tensor_org'])
    #     pass

    # for x in dataset_train:
    #     display_bw_batch_pyplot(x['tensor_bw'])
    #     display_ycbcr_batch_pyplot(x['tensor_org'])
    #     pass

    return dataset_train, dataset_validation
