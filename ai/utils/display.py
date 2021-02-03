import tensorflow_io as tfio
import matplotlib.pyplot as plt

from .image_processing import tensor2image


def display_ycbcr_tensor_pyplot(tensor):
    image_rgb = tfio.experimental.color.ycbcr_to_rgb(tensor2image(tensor))
    plt.axis('off')
    plt.imshow(image_rgb)
    plt.show()


def display_bw_tensor_pyplot(tensor):
    image = tensor2image(tensor)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()


def display_ycbcr_batch_pyplot(batch):
    for tensor in batch:
        display_ycbcr_tensor_pyplot(tensor)


def display_bw_batch_pyplot(batch):
    for tensor in batch:
        display_bw_tensor_pyplot(tensor)
