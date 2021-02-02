from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.convolutional import Conv2D


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_1 = Conv2D(64, 3, activation='relu', padding='same')
        self.upsampling_1 = UpSampling2D((2, 2))
        self.conv_2 = Conv2D(64, 3, activation='relu', padding='same')
        self.upsampling_2 = UpSampling2D((2, 2))
        self.conv_3 = Conv2D(1, 3, activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.upsampling_1(x)
        x = self.conv_2(x)
        x = self.upsampling_2(x)
        return self.conv_3(x)
