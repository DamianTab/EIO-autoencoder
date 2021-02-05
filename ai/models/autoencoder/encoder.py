from tensorflow.keras import *
from tensorflow.keras.layers import *


class Encoder(Model):
    def __init__(self, batch_normalization, dropout_rate, l2_regularization, momentum):
        super(Encoder, self).__init__()
        self.batch_normalization_1 = batch_normalization

        self.conv_1 = Conv2D(16, 3, strides=(2, 2), activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(l2_regularization),
                             bias_regularizer=regularizers.l2(l2_regularization))
        if self.batch_normalization:
            self.batch_norm_1 = BatchNormalization(momentum)
        self.conv_2 = Conv2D(32, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(l2_regularization),
                             bias_regularizer=regularizers.l2(l2_regularization))
        self.dropout_1 = Dropout(dropout_rate)
        self.conv_3 = Conv2D(64, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(l2_regularization),
                             bias_regularizer=regularizers.l2(l2_regularization))

    def call(self, inputs):
        x = self.conv_1(inputs)
        if self.batch_normalization:
            x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = self.dropout_1(x)
        x = self.conv_3(x)
        return x
