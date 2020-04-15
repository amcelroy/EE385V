import numpy as np
import keras
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ELU, Dropout, ReLU


class STFTNet:
    def __init__(self):
        self.__model__ = keras.Model

    def down(self, input, neurons=32):
        conv = Conv3D(
            filters=neurons,
            kernel_size=(3, 3, 3),
            padding='same')(input)
        conv = ReLU()(conv)
        conv = Dropout(0)(conv)
        down_sample = MaxPooling3D(pool_size=(1, 1, 2), strides=(1, 1, 2))(conv)
        return down_sample

    def up(self, input, neurons=32):
        up_sample = UpSampling3D(size=(1, 1, 2))(input)
        conv = Conv3D(
            filters=neurons,
            kernel_size=(3, 3, 3),
            padding='same')(up_sample)
        conv = ReLU()(conv)
        return conv

    def init(self, input_shape=(1, 2, 3)):
        input = Input(batch_shape=(None,) + tuple(input_shape))

        down1 = self.down(input, 16)
        up1 = self.up(down1, 16)

        output = Conv3D(
            filters=1,
            kernel_size=(3, 3, 3),
            padding='same',
        )(up1)
        output = ELU()(output)

        self.__model__ = Model(inputs=input, outputs=output)

        return self.__model__

    def compile(self):
        opt = keras.optimizers.sgd(
            learning_rate=.001,
            nesterov=True,
            momentum=.01,
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