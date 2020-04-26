import numpy as np
import keras
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, ELU, Dropout, ReLU, MaxPooling2D, UpSampling2D, Conv2D, \
    LocallyConnected2D, SeparableConv2D, Dense, Flatten, Reshape, Conv1D, MaxPooling1D, UpSampling1D


class PowerNet:
    def __init__(self):
        self.__model__ = keras.Model

    def down(self, input, neurons=32):
        conv = Conv1D(
            filters=neurons,
            kernel_regularizer=keras.regularizers.l2(.1),
            kernel_size=3,
        )(input)
        conv = ReLU()(conv)
        conv = MaxPooling1D(2)(conv)
        #conv = Dropout(.2)(conv)
        return conv

    def init(self, input_shape=(1, 2, 3)):
        input = Input(batch_shape=(None,) + tuple(input_shape))

        # d1 = self.down(input, input_shape[-1]*2)
        # d1 = self.down(d1, input_shape[-1] * 4)
        # d1 = self.down(d1, input_shape[-1] * 8)

        flat = Flatten()(input)

        f1 = Dense(256, activation='relu')(flat)
        drop = Dropout(.1)(f1)
        f1 = Dense(256, activation='relu')(drop)
        drop = Dropout(.1)(f1)
        # f1 = Dense(32, activation='relu')(drop)
        # drop = Dropout(0)(f1)

        f2 = Dense(2, activation='softmax')(drop)

        self.__model__ = Model(inputs=input, outputs=f2)

        print(self.__model__.summary())

        return self.__model__

    def compile(self):
        opt = keras.optimizers.adam(
            learning_rate=.0001,
        )
        self.__model__.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
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