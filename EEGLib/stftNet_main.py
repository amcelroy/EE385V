import os
from configparser import ConfigParser
import numpy as np
from scipy.stats import zscore

import stftNet
from DatasetLoader import DatasetLoader
from EEG import EEG
import matplotlib.pyplot as plt

from Feature.STFTFeature import STFTFeature

stftn = stftNet.STFTNet()

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

root = parser['DATA']['root']

dsl = DatasetLoader(root)

offline_dict = dsl.getOffline()

error_master = []
no_error_master = []

channels_of_interest = ['FCz', 'CZ', 'FC1', 'FC2', 'CPz', 'C4', 'CP4']

# Create or load the data
if not (os.path.exists('offline_error.npy') and os.path.exists('offline_no_error.npy')):
    for subject in offline_dict.keys():
        runs = offline_dict[subject]

        count = 0
        for run in runs:
            path = os.path.join(root, subject, 'Offline', run)
            e = EEG()
            (eeg, trig, mat) = e.open(
                path,
                channel_names=channel_names_spellers
            )

            # theta = e.getTheta()
            theta = eeg

            error = e.getEEGTrials(
                pretime=.5,
                posttime=1,
                error=True,
                plot=False,
                gdf=theta,
                channels=channels_of_interest,
            )

            no_error = e.getEEGTrials(
                pretime=.5,
                posttime=1,
                error=False,
                plot=False,
                gdf=theta,
                channels=channels_of_interest,
            )

            error = e.CARFilter(error)
            no_error = e.CARFilter(no_error)

            stftfeature = STFTFeature()

            error, grand_avg_error, grand_var_error, freq, time = stftfeature.extract(error,
                                                                                window=512,
                                                                                overlap=496,
                                                                                pre_trigger_time=.5,
                                                                                fs=512,
                                                                                frequency_range=[4, 9])

            no_error, grand_avg_no_error, grand_var_no_error, freq, time = stftfeature.extract(no_error,
                                                                                window=512,
                                                                                overlap=496,
                                                                                pre_trigger_time=.5,
                                                                                fs=512,
                                                                                frequency_range=[4, 9])

            if error_master == []:
                error_master = error
            else:
                error_master = np.append(error_master, error, axis=0)

            if no_error_master == []:
                no_error_master = no_error
            else:
                no_error_master = np.append(no_error_master, no_error, axis=0)

    np.save('offline_error', error_master)
    np.save('offline_no_error', no_error_master)
else:
    error_master = np.load('offline_error.npy')
    no_error_master = np.load('offline_no_error.npy')

error_master = error_master[..., 1:]
error_master = np.expand_dims(error_master, 4)

no_error_master = no_error_master[..., 1:]
no_error_master = np.expand_dims(no_error_master, 4)

error_master = zscore(error_master)
no_error_master = zscore(no_error_master)

# Split the error data into 80/20 test/train
indecies = np.arange(error_master.shape[0])
np.random.shuffle(indecies)

validation_index = indecies[0:no_error_master.shape[0]]
train_index = indecies[no_error_master.shape[0]:]

train_set = error_master[train_index, ...]
val_set = error_master[validation_index, ...]

snet = stftNet.STFTNet()
snet.init(train_set.shape[1:])
model = snet.compile()

model.fit(train_set, train_set,
          batch_size=32,
          epochs=1000,
          verbose=2,
          shuffle=True,
          validation_data=(no_error_master, no_error_master),
          )

x = 0


