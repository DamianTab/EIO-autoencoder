import tensorflow as tf
import tensorflow_io as tfio


# @tf.function
def prepare_image_as_input(x):
    image = x['image']

    tensor_ycbcr = tfio.experimental.color.rgb_to_ycbcr(image)
    tensor_ycbcr = tf.cast(tensor_ycbcr, tf.float32) - 127.0
    tensor_ycbcr /= 127.0

    tensor_bw = tfio.experimental.color.rgb_to_grayscale(image)
    tensor_bw -= 127
    tensor_bw = tf.cast(tensor_bw, tf.float32) / 127.0

    return {
        'tensor_org': tensor_ycbcr,  # Original colorful image as tensor
        'tensor_bw': tensor_bw  # Black and white image as tensor
    }


def prepare_image_as_input2(x, y):
    image = x

    tensor_ycbcr = tf.cast(image, tf.uint8)
    tensor_ycbcr = tfio.experimental.color.rgb_to_ycbcr(tensor_ycbcr)
    tensor_ycbcr = tf.cast(tensor_ycbcr, tf.float32) - 127.0
    tensor_ycbcr /= 127

    tensor_bw = tfio.experimental.color.rgb_to_grayscale(image)
    tensor_bw -= 127
    tensor_bw = tf.cast(tensor_bw, tf.float32) / 127.0

    return {
        'tensor_org': tensor_ycbcr,  # Original colorful image as tensor
        'tensor_bw': tensor_bw  # Black and white image as tensor
    }


def tensor2image(img):
    img *= 127
    img += 127
    return tf.cast(img, tf.uint8)
