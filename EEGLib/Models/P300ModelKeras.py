import keras
import numpy as np

from Models.BCIModel import BCIModel

l2reg = 1
gnoise = .1

class P300ModelKeras(BCIModel):
    def __init__(self):
        self.__model = keras.Model

    def downlayer(self, input, kernel_size=[3,3], kernels=32):
        l1 = keras.layers.Conv2D(filters=kernels,
                                 kernel_size=kernel_size,
                                 kernel_regularizer=keras.regularizers.l2(l2reg)
                                 )(input)
        l2 = keras.layers.GaussianNoise(gnoise)(l1)
        l3 = keras.layers.ELU()(l2)




    def createModel(self, input_shape=[16, 32, 32]):
        super().createModel()
        input_layer = keras.layers.Input(shape=input_shape, )




    def fit(self, x=np.Array, y=np.Array):
        super().fit(x, y)

    def predict(self, x=np.Array):
        super().predict(x)