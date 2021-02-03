from tensorflow.keras import *
from tensorflow.keras.layers import *


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_1 = Conv2D(32, 3, activation='relu', padding='same')
        self.upsampling_1 = UpSampling2D((2, 2))

        self.conv_2 = Conv2D(16, 3, activation='relu', padding='same')
        self.upsampling_2 = UpSampling2D((2, 2))

        self.conv_3 = Conv2D(8, 3, activation='relu', padding='same')
        self.upsampling_3 = UpSampling2D((2, 2))

        self.conv_out = Conv2D(3, 3, activation='tanh', padding='same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.upsampling_1(x)
        x = self.conv_2(x)
        x = self.upsampling_2(x)
        x = self.conv_3(x)
        x = self.upsampling_3(x)
        x = self.conv_out(x)
        return x
