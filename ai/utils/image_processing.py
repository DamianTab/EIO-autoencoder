import tensorflow as tf
import tensorflow_io as tfio


@tf.function
def prepare_image_as_input(x, y):  # For datasets loaded from image files
    image = x

    image_ycbcr = tf.cast(image, tf.uint8)
    image_ycbcr = tfio.experimental.color.rgb_to_ycbcr(image_ycbcr)
    image_ycbcr = tf.cast(image_ycbcr, tf.float32) - 127.0
    image_ycbcr /= 127

    return image_ycbcr


def scale_image(img):
    img *= 127
    img += 127
    return tf.cast(img, tf.uint8)
