from tensorflow.python.keras.models import Model

from ai.models.decoder import Decoder
from ai.models.encoder import Encoder


class AutoEncoder(Model):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)