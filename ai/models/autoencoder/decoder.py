from tensorflow.keras import *
from tensorflow.keras.layers import *


class Decoder(Model):
    def __init__(self, batch_normalization, dropout_rate, l2_regularization, momentum):
        super(Decoder, self).__init__()
        self.batch_normalization = batch_normalization
        if self.batch_normalization:
            self.batch_norm_1 = BatchNormalization(momentum=momentum)

        self.conv_1 = Conv2D(64, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(l2_regularization),
                             bias_regularizer=regularizers.l2(l2_regularization))

        self.conv_2 = Conv2D(32, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(l2_regularization),
                             bias_regularizer=regularizers.l2(l2_regularization))

        self.dropout_1 = Dropout(dropout_rate)
        self.conv_3 = Conv2D(16, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(l2_regularization),
                             bias_regularizer=regularizers.l2(l2_regularization))
        self.upsampling_1 = UpSampling2D((2, 2))

        self.conv_out = Conv2D(3, 3, activation='tanh', padding='same',
                               kernel_regularizer=regularizers.l2(l2_regularization),
                               bias_regularizer=regularizers.l2(l2_regularization))

    def call(self, inputs):
        x = self.conv_1(inputs)
        if self.batch_normalization:
            x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        x = self.conv_3(x)
        x = self.upsampling_1(x)

        x = self.conv_out(x)
        return x
