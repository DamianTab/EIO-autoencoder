from tensorflow.keras import *

from ai.models.decoder import Decoder
from ai.models.encoder import Encoder


class AutoEncoder(Model):

    def __init__(self, batch_normalization=False, dropout_rate=0, l2_regularization=0):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(batch_normalization, dropout_rate, l2_regularization)
        self.decoder = Decoder(batch_normalization, dropout_rate, l2_regularization)

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

