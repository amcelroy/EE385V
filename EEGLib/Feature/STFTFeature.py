import numpy as np
import matplotlib as plt
from celluloid import Camera
from scipy.signal import stft

from EEGLib.Feature.BCIFeature import BCIFeature


class STFTFeature(BCIFeature):

    def extract(self, eegVolume=np.ndarray, window=64, overlap=48, plot=False, trigger_event=[0],
             fs=512, pre_trigger_time=0, save_plot_filename='', ac_filter=False):
        '''
        Wraps Scipy STFT and applies it to an EEG Volume

        :param eegVolume: 3D array of (Trigger Event, EEG, Time Samples)
        :param window_size: FFT Size
        :param overlap: Overlaps, in pixels for the FFT
        :param fs: Sampling frequency, defaults to 512 samples per second
        :param pre_trigger_time Time, in seconds, of samples contained in the pretrigger
        :return: Tuple:
                    4D Array of Data (Trigger Event, EEG, Spectragram Freq, Spectragram Time)
                    1D Frequency
                    1D Time
        '''
        super().extract(window)
        freq, time, data = stft(eegVolume, fs=fs, nperseg=window, noverlap=overlap, axis=-1)
        data = np.abs(data)
        data = data[:, :, :16, :]

        if ac_filter:
            data[:, :, 0, :] = 0

        time -= pre_trigger_time

        grand_avg = np.mean(data, axis=0)
        grand_var = np.var(data, axis=0)

        if plot:
            fig, ax = plt.subplots(16, 2, figsize=(20, 10))
            camera = Camera(fig)
            for t in trigger_event[0]:
                time_label = ['{:.2f}'.format(x) for x in time]
                slice = data[t, ...]

                for x, y in np.ndindex(ax.shape):
                    if y == 0:
                        ax[x, y].imshow(slice[x], aspect="auto")
                    elif y == 1:
                        ax[x, y].imshow(slice[x] - grand_avg[x], aspect='auto')

                    ax[x, y].xaxis.set_visible(False)
                    ax[x, y].yaxis.set_visible(False)
                ax[x, y].xaxis.set_visible(True)
                ax[x, y].set_xticks(np.arange(len(time_label)))
                ax[x, y].set_xticklabels(time_label, rotation=60)
                camera.snap()

            animation = camera.animate()
            fig.show()

            if save_plot_filename != '':
                plt.savefig(save_plot_filename)

        return data, freq, time, grand_avg, grand_var
