import numpy as np
import keras
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ELU, Dropout, ReLU, MaxPooling2D, UpSampling2D, Conv2D, \
    LocallyConnected2D, SeparableConv2D, Dense, Flatten, Reshape


class STFTNet:
    def __init__(self):
        self.__model__ = keras.Model

    def down(self, input, neurons=32):
        conv = SeparableConv2D(
            filters=neurons,
            kernel_size=(3, 3),
            padding='same')(input)
        conv = ReLU()(conv)
        conv = Dropout(.2)(conv)
        down_sample = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(conv)
        return down_sample

    def up(self, input, neurons=32):
        up_sample = UpSampling2D(size=(1, 2))(input)
        conv = SeparableConv2D(
            filters=neurons,
            kernel_size=(3, 3),
            padding='same')(up_sample)
        conv = ReLU()(conv)
        conv = Dropout(.2)(conv)
        return conv

    def init(self, input_shape=(1, 2, 3)):
        input = Input(batch_shape=(None,) + tuple(input_shape))

        down1 = self.down(input, 12)
        down2 = self.down(down1, 24)
        down3 = self.down(down2, 48)

        flat = Flatten()(down3)

        drop = Dropout(.5)(flat)
        f1 = Dense(1440, activation='relu')(drop)
        drop = Dropout(.5)(f1)
        f2 = Dense(1440, activation='relu')(drop)

        reshape = Reshape((5, 6, 48))(f2)

        up3 = self.up(reshape, 48)
        up2 = self.up(up3, 24)
        up1 = self.up(up2, 12)

        output = SeparableConv2D(
            filters=6,
            kernel_size=(3, 3),
            padding='same',
        )(up1)
        output = ReLU()(output)

        self.__model__ = Model(inputs=input, outputs=output)

        return self.__model__

    def compile(self):
        opt = keras.optimizers.adamax(
            learning_rate=.001,
        )
        self.__model__.compile(
            optimizer=opt,
            loss='mean_squared_error'
        )
        return self.__model__

    def train_offline(self):
        pass

    def test_s2(self, s2=np.ndarray):
        pass

    def continue_train_s2(self, s2=np.ndarray):
        pass

    def test_s3(self, s3=np.ndarray):
        pass