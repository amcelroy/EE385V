import os

from DatasetLoader import DatasetLoader
from EEG import EEG
import numpy as np
import matplotlib.pyplot as plt
from Feature.STFTFeature import STFTFeature
from configparser import ConfigParser

channel_names_spellers = {
    'eeg:1': 'Fz',
    'eeg:2': 'FC3',
    'eeg:3': 'FC1',
    'eeg:4': 'FCz',
    'eeg:5': 'FC2',
    'eeg:6': 'FC4',
    'eeg:7': 'C3',
    'eeg:8': 'C1',
    'eeg:9': 'Cz',
    'eeg:10': 'C2',
    'eeg:11': 'C4',
    'eeg:12': 'CP3',
    'eeg:13': 'CP1',
    'eeg:14': 'CPz',
    'eeg:15': 'CP2',
    'eeg:16': 'CP4'
}

parser = ConfigParser()
parser.read('config.ini')

root = parser['INFO']['root']

dsl = DatasetLoader(root)

offline_dict = dsl.getOffline()

fig_size = (12, 7)

for subject in offline_dict.keys():
    runs = offline_dict[subject]

    fig, ax = plt.subplots(len(channel_names_spellers), len(offline_dict.keys()),\
         figsize = fig_size)

    count = 0
    for run in runs:
        path = os.path.join(root, subject, 'Offline', run)
        e = EEG()
        (eeg, trig, mat) = e.open(
            path,
            channel_names=channel_names_spellers
        )

        theta = e.getTheta()

        error = e.getEEGTrials(
            pretime=.5,
            posttime=1,
            error=True,
            plot=False,
            gdf=theta,
        )
        no_error = e.getEEGTrials(pretime=.5, posttime=1, error=False, plot=False, gdf=theta)

        error = e.CARFilter(error)
        no_error = e.CARFilter(no_error)

        averaged_error = np.mean(error, axis=0)
        averaged_error = e.addOffset(averaged_error, 1e-6)

        averaged_no_Error = np.mean(no_error, axis=0)
        averaged_no_Error = e.addOffset(averaged_no_Error, 1e-6)

        stftfeature = STFTFeature()
        spectro, freq, time, grand_avg, grand_var = stftfeature.extract(error,
                                                                        plot=False,
                                                                        window=512,
                                                                        overlap=468,
                                                                        trigger_event=[20],
                                                                        pre_trigger_time=.5,
                                                                        fs=512,
                                                                        frequency_range=[4, 8],
                                                                        channel_names=e.getChannelNames())

        stftfeature.plot(ax[..., count], grand_avg)
        if not count:
            stftfeature.addYLabels(ax[..., count], e.getChannelNames())
        count += 1

    plt.title('{} identified as Error Potentials'.format(subject))
    plt.show()
    x = 0