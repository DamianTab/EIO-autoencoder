import matplotlib.pyplot as plt
import tensorflow_io as tfio

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


def display_compare_results_pyplot(originals, predictions, display_count=3):
    for i in range(display_count):
        fix, (ax1, ax2) = plt.subplots(1, 2)
        orig_rgb = tfio.experimental.color.ycbcr_to_rgb(tensor2image(originals[i]))
        pred_rgb = tfio.experimental.color.ycbcr_to_rgb(tensor2image(predictions[i]))
        ax1.axis('off')
        ax1.set_title('Expected output')
        ax1.imshow(orig_rgb)
        ax2.axis('off')
        ax2.set_title('Actual output')
        ax2.imshow(pred_rgb)
        plt.show()


def display_compare_results_pyplot2(inputs, originals, predictions, display_count=3):
    for i in range(display_count):
        fix, (ax1, ax2, ax3) = plt.subplots(1, 3)
        input_bw = tensor2image(inputs[i])
        orig_rgb = tfio.experimental.color.ycbcr_to_rgb(tensor2image(originals[i]))
        pred_rgb = tfio.experimental.color.ycbcr_to_rgb(tensor2image(predictions[i]))
        ax1.axis('off')
        ax1.set_title('Input')
        ax1.imshow(input_bw, cmap='gray')
        ax2.axis('off')
        ax2.set_title('Expected output')
        ax2.imshow(orig_rgb)
        ax3.axis('off')
        ax3.set_title('Actual output')
        ax3.imshow(pred_rgb)
        plt.show()
