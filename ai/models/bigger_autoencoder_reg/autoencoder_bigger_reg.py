from tensorflow.keras import *

from .decoder_bigger_reg import BiggerDecoderReg
from .encoder_bigger_reg import BiggerEncoderReg


class BiggerAutoEncoderReg(Model):

    def __init__(self, batch_normalization=False, dropout_rate=0, l2_regularization=0, momentum=0):
        super(BiggerAutoEncoderReg, self).__init__()
        self.encoder = BiggerEncoderReg(batch_normalization, dropout_rate, l2_regularization, momentum)
        self.decoder = BiggerDecoderReg(batch_normalization, dropout_rate, l2_regularization, momentum)

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

