from tensorflow.keras import *
from tensorflow.keras.layers import *


class BiggerDecoder(Model):
    def __init__(self):
        super(BiggerDecoder, self).__init__()
        self.conv_1 = Conv2DTranspose(128, 3, activation='relu', padding='same')
        self.upsampling_1 = UpSampling2D((2, 2))
        # self.d1 = Dropout(0.5)

        self.conv_2 = Conv2DTranspose(64, 3, activation='relu', padding='same')
        self.upsampling_2 = UpSampling2D((2, 2))
        # self.d2 = Dropout(0.5)

        self.conv_3 = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.upsampling_3 = UpSampling2D((2, 2))
        # self.d3 = Dropout(0.5)

        self.conv_4 = Conv2DTranspose(16, 3, activation='relu', padding='same')
        self.upsampling_4 = UpSampling2D((2, 2))
        # self.d4 = Dropout(0.5)

        self.conv_out = Conv2D(3, 3, activation='tanh', padding='same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.upsampling_1(x)
        # x = self.d1(x)
        x = self.conv_2(x)
        x = self.upsampling_2(x)
        # x = self.d2(x)
        x = self.conv_3(x)
        x = self.upsampling_3(x)
        # x = self.d3(x)
        x = self.conv_4(x)
        x = self.upsampling_4(x)
        # x = self.d4(x)
        x = self.conv_out(x)
        return x
