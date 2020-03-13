import numpy as np

from Feature.BCIFeature import BCIFeature


class NormalFeature(BCIFeature):

    def extract(self, eegVolume=np.ndarray, window=32, overlap=31):
        super().extract(eegVolume, window)

        if window - overlap < 0:
            IndexError('Window must be larger than overlap')

        dt = window - overlap
        out_array = []
        for x in range(0, eegVolume.shape[2], dt):
            sub = eegVolume[..., x:(x + window)]
            sub = np.mean(sub, axis=-1)
            out_array.append(sub)
        power_array = np.array(out_array)
        power_array = np.swapaxes(power_array, 0, -1)

        grand_avg = np.mean(power_array, axis=-1)
        grand_var = np.var(power_array, axis=-1)

        return power_array, grand_avg, grand_var
