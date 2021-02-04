from tensorflow.keras import *
from tensorflow.keras.layers import *


class BiggerEncoder(Model):
    def __init__(self):
        super(BiggerEncoder, self).__init__()
        self.conv_1 = Conv2D(16, 3, strides=(2, 2), activation='relu', padding='same')
        self.conv_2 = Conv2D(32, 3, strides=(2, 2), activation='relu', padding='same')
        self.conv_3 = Conv2D(64, 3, strides=(2, 2), activation='relu', padding='same')
        self.conv_4 = Conv2D(128, 3, strides=(2, 2), activation='relu', padding='same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x
