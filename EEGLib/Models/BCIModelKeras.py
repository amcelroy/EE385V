from EEGLib.BCIModel import BCIModel
import numpy as np


class BCIModelKeras(BCIModel):
    def createModel(self):
        super().createModel()

    def fit(self, x=np.Array, y=np.Array):
        super().fit(x, y)

    def predict(self, x=np.Array):
        super().predict(x)