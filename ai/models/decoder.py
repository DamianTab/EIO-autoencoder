from tensorflow.keras import *
from tensorflow.keras.layers import *


class Decoder(Model):
    def __init__(self, batch_normalization, dropout_rate, l2_regularization):
        super(Decoder, self).__init__()
        self.batch_normalization = batch_normalization
        self.dropout = Dropout(dropout_rate)
        self.batch_norm = BatchNormalization()

        self.conv_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_regularization))
        self.upsampling_1 = UpSampling2D((2, 2))

        self.conv_2 = Conv2D(16, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_regularization))
        self.upsampling_2 = UpSampling2D((2, 2))

        self.conv_3 = Conv2D(8, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_regularization))
        self.upsampling_3 = UpSampling2D((2, 2))

        self.conv_out = Conv2D(2, 3, activation='tanh', padding='same', kernel_regularizer=regularizers.l2(l2_regularization))

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.upsampling_1(x)

        x = self.conv_2(x)
        x = self.upsampling_2(x)

        x = self.dropout(x)
        x = self.conv_3(x)
        x = self.upsampling_3(x)

        if self.batch_normalization:
            x = self.batch_norm(x)
        x = self.conv_out(x)
        return x
