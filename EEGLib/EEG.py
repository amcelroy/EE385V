import mne
import scipy
from scipy import io
from mne import Epochs
from mne.channels import Layout
import numpy as np
import os
from pathlib import Path

from EEGLib.EE385VMatFile import EE385VMatFile


class EEG:
    def __init__(self):
        self.__gdf = mne.io.Raw
        self.__mat = EE385VMatFile
        self.__eeg = mne.io.Raw
        self.__trigger = mne.io.Raw

    def open(self, filepath):
        '''
        @param filepath: Path to GDF or .MAT file
        @:return Tuple (EEG data scaled to V, trigger, EE385VMatFile)
        '''

        path = Path(filepath)

        gdf_path = os.path.basename(filepath)
        mat_path = os.path.basename(filepath)
        if path.suffix == '.gdf':
            gdf_path = filepath
            mat_path = str(path.parent) + '/' + str(path.stem) + '.mat'
        elif path.suffix == '.mat':
            mat_path = filepath
            gdf_path = str(path.parent) + '/' + str(path.stem) + '.gdf'
        else:
            raise IOError('Filepath needs to be either a .gdf or .mat file')

        gdf = mne.io.read_raw_gdf(gdf_path, preload=True)
        self.__gdf = gdf
        # Scale to microvolts, see: https://github.com/mne-tools/mne-python/issues/5539
        self.__eeg, self.__trigger = self.splitTrigger(trigger='trigger')
        self.__eeg = self.__eeg.apply_function(lambda x: x / 1e6)
        self.__mat = EE385VMatFile(mat_path)
        return (self.__eeg, self.__trigger, self.__mat)

    def getChannelNames(self):
        '''
        :return: the channel names as a list of strings
        '''
        return self.__gdf.ch_names

    def printChannels(self):
        '''
        Convenience function to print the channel names
        :return:
        '''
        print('Availible Channels: ' + str(self.getChannelNames()))

    def getRawEEG(self, channels=['']):
        '''
        Packages https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data

        :param channels: String list of EEG channels to fetch
        :return: Tuple of ([EEG Channel, Data], Time)
        '''
        if channels == ['']:
            return self.__gdf.get_data(return_times=True)
        else:
            return self.__gdf.get_data(picks=channels, return_times=True)

    def splitTrigger(self, trigger='trigger'):
        '''
        Splits the trigger from the EEG signal

        :param trigger: Name of the trigger. i.e. trigger if the channel is trigger:1
        :return: Tuple (Data without Trigger, Trigger)
        '''
        for channels in self.getChannelNames():
            if trigger in channels:
                trigger = self.__gdf.get_data(picks=channels)
                data_no_trigger = self.__gdf.drop_channels(channels)
                return (data_no_trigger, trigger)


    def getAlpha(self):
        '''
        Filters the data for the alpha range of frequencies

        :return:
        '''
        return self.__gdf.filter(l_freq=8, h_freq=12)

    def getTheta(self):
        return self.__gdf.filter(l_freq=1, h_freq=8)

    def getBeta(self):
        return self.__gdf.filter(l_freq=12, h_freq=30)

    def timeToSample(self, time, fs=512):
        return time*fs

    def timeArrayToSampleArray(self, time=np.ndarray, fs=512):
        return np.apply_along_axis(self.timeToSample, 0, time, fs=fs).astype(np.uint)

    def identifyErrors(self):
        annotation_time = self.__gdf.annotations.onset[1:]
        annotation_time = annotation_time[:-1]
        annotation_desc = self.__gdf.annotations.description[1:]
        annotation_desc = annotation_desc[:-1]
        intention = self.__mat.actions()
        states = self.__mat.states()[:, 1:]
        states = states[:, :len(intention)]
        error = []
        for x in range(states.shape[1]):
            if states[0, x] != states[1, x]:
                error.append(True)
            else:
                error.append(False)
        return (np.asarray(error, dtype=np.bool_), annotation_time)


if __name__ == "__main__":
    e = EEG()
    (eeg, trig, mat) = e.open(
        '/home/amcelroy/Code/EE385V/BCI Course 2020/ErrPSpeller/Subject1/Offline/ad4_raser_offline_offline_171110_172431.gdf')

    error, time = e.identifyErrors()
    time_as_sample = e.timeArrayToSampleArray(time)

    e.printChannels()
    # alpha = e.getAlpha()
    # alpha.plot(decim=1)

    theta = e.getTheta()
    theta.plot(decim=1)
    #gdf.printChannels()
    channels = e.getRawEEG()
    mat.targetLetter()
    mat.ringBuffer()
    #print(gdf.info)
