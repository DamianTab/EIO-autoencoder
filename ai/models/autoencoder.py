from tensorflow.keras import *

from ai.models.decoder import Decoder
from ai.models.encoder import Encoder


class AutoEncoder(Model):

    def __init__(self, batch_normalization=False, dropout=0, l2_regularization=0):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

