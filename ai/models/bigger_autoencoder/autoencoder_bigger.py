from tensorflow.keras import *

from .decoder_bigger import BiggerDecoder
from .encoder_bigger import BiggerEncoder


class BiggerAutoEncoder(Model):

    def __init__(self, batch_normalization=False, dropout_rate=0, l2_regularization=0, momentum=0):
        super(BiggerAutoEncoder, self).__init__()
        self.encoder = BiggerEncoder()
        self.decoder = BiggerDecoder()

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

