from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.convolutional import Conv2D


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_1 = Conv2D(64, 3, activation='relu', padding='same')
        self.pooling_1 = MaxPooling2D((2, 2), padding='same')
        self.conv_2 = Conv2D(64, 3, activation='relu', padding='same')
        self.pooling_2 = MaxPooling2D((2, 2), padding='same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.pooling_1(x)
        x = self.conv_2(x)
        return self.pooling_2(x)
