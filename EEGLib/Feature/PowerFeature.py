import numpy as np

from EEGLib.Feature.BCIFeature import BCIFeature


class PowerFeature(BCIFeature):

    def extract(self, eegVolume=np.ndarray, window=64, overlap=48):
        super().extract(eegVolume, window)
        power = np.square(eegVolume)

        if window - overlap < 0:
            IndexError('Window must be larger than overlap')

        dt = window - overlap
        #for x in range(eegVolume.shape[2]):

