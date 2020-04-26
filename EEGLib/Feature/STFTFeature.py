import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from scipy.signal import stft

from Feature.BCIFeature import BCIFeature


class STFTFeature(BCIFeature):

    def extract(self, eegVolume=np.ndarray, window=64, overlap=48, plot=False, trigger_event=[0],
                fs=512, pre_trigger_time=0, save_plot_filename='', frequency_range=None, channel_names=None):
        '''
        Wraps Scipy STFT and applies it to an EEG Volume

        :param eegVolume: 3D array of (Trigger Event, EEG, Time Samples)
        :param window_size: FFT Size
        :param overlap: Overlaps, in pixels for the FFT
        :param fs: Sampling frequency, defaults to 512 samples per second
        :param frequency_range: Start and stop frequency to view. Note, is fs=512 and window is 64, each frequency
        unit in the stft will fs/window_size (512/64).
        :param pre_trigger_time Time, in seconds, of samples contained in the pretrigger
        :return: Tuple:
                    4D Array of Data (Trigger Event, EEG, Spectragram Freq, Spectragram Time)
                    1D Frequency
                    1D Time
        '''
        super().extract(window)
        freq, time, data = stft(eegVolume, fs=fs, nperseg=window, noverlap=overlap, axis=-1)
        data = np.abs(data)
        data = data[:, :16, :]

        if frequency_range:
            data = data[:, frequency_range[0]:frequency_range[1], :]

        time -= pre_trigger_time

        # grand_avg = np.mean(data, axis=-1)
        # grand_var = np.var(data, axis=-1)
        #
        # if plot:
        #     fig, ax = plt.subplots(16, 2, figsize=(20, 10))
        #     camera = Camera(fig)
        #     for t in trigger_event:
        #         time_label = ['{:.2f}'.format(x) for x in time]
        #         slice = data[t, ...]
        #
        #         for x, y in np.ndindex(ax.shape):
        #             if y == 0:
        #                 ax[x, y].imshow(slice[x], aspect="auto")
        #             elif y == 1:
        #                 ax[x, y].imshow(grand_avg[x], aspect='auto')
        #
        #             if channel_names:
        #                 ax[x, y].xaxis.set_visible(False)
        #                 ax[x, 0].yaxis.set_visible(True)
        #                 ax[x, 0].set_ylabel(channel_names[x])
        #                 ax[x, 0].set_yticks([])
        #                 ax[x, 1].set_yticks([])
        #             else:
        #                 ax[x, y].xaxis.set_visible(False)
        #                 ax[x, y].yaxis.set_visible(False)
        #         ax[x, 0].xaxis.set_visible(True)
        #         ax[x, 0].set_xticks(np.arange(len(time_label)))
        #         ax[x, 0].set_xticklabels(time_label, rotation=60)
        #         ax[x, 1].xaxis.set_visible(True)
        #         ax[x, 1].set_xticks(np.arange(len(time_label)))
        #         ax[x, 1].set_xticklabels(time_label, rotation=60)
        #         camera.snap()
        #
        #     animation = camera.animate()
        #     fig.show()
        #
        #     if save_plot_filename != '':
        #         plt.savefig(save_plot_filename)

        return data, freq, time

    def plot(self, ax=plt.Axes, data=np.ndarray, zero_time=48):
        for x in range(data.shape[0]):
            ax[x].imshow(data[x], aspect='auto')
            ax[x].xaxis.set_visible(False)
            ax[x].yaxis.set_visible(False)
            p1 = [zero_time, 0]
            p2 = [zero_time, data.shape[1] - 1]
            ax[x].plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')

    def addYLabels(self, ax=plt.Axes, channels=['']):
        for x in range(len(channels)):
            ax[x].set_ylabel(channels[x])
            ax[x].set_yticklabels([])
            ax[x].yaxis.set_visible(True)

    def addFreqLabels(self, ax=plt.Axes, tick_labels=['']):
        for x in range(ax.shape[0]):
            ax[x].set_yticks(np.arange(len(tick_labels)))
            ax[x].set_yticklabels(tick_labels)
            ax[x].set_ylabel('Hz')
            ax[x].yaxis.set_visible(True)
            ax[x].yaxis.tick_right()
            ax[x].yaxis.set_label_position("right")

    def addXLabels(self, ax=plt.Axes, time=[]):
        time_label = ['{:.2f}'.format(x) for x in time]
        ax[ax.shape[0] - 1].xaxis.set_visible(True)
        ax[ax.shape[0] - 1].set_xticks(np.arange(len(time_label)))
        ax[ax.shape[0] - 1].set_xticklabels(time_label, rotation=90)
        ax[ax.shape[0] - 1].set_xlabel('Time in seconds')

    def addTitle(self, ax=plt.Axes, title=''):
        ax[0].set_title(title)
