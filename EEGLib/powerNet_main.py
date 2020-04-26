import os
from configparser import ConfigParser
import numpy as np
from keras.callbacks import Callback
from scipy.stats import zscore

import stftNet
from DatasetLoader import DatasetLoader
from EEG import EEG
import matplotlib.pyplot as plt

from Feature.NormalFeature import NormalFeature
from Feature.PowerFeature import PowerFeature
from Feature.STFTFeature import STFTFeature
from powerNet import PowerNet

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

# Get dictionaries with filenames for each dataset
offline_dict = dsl.getTrials()[0]
S2_dict = dsl.getTrials()[1]
S3_dict = dsl.getTrials()[2]

channels_of_interest = ['FCz', 'C4', 'Cz', 'FC1', 'FC2', 'CPz', 'CP4']


# adding each trial with its own variable for error_master, no_error_master
def trials(trial_name=''):
    error_master = []
    no_error_master = []
    trial_dicts = {
        'Offline': offline_dict,
        'S2': S2_dict,
        'S3': S3_dict,
    }
    # Create or load the data
    if not (os.path.exists(f'{trial_name}_error.npy') and os.path.exists(f'{trial_name}_no_error.npy')):
        trial = trial_dicts[trial_name]
        for subject in trial.keys():
            runs = trial[subject]
            #                count = 0
            for run in runs:
                path = os.path.join(root, subject, trial_name, run)
                e = EEG()
                (eeg, trig, mat) = e.open(
                    path,
                    channel_names=channel_names_spellers
                )

                #theta = e.getTheta()
                theta = e.getRawEEG(channels=channels_of_interest)[0]
                theta = e.CARFilter(theta)

                powerFeature = PowerFeature()
                theta, grand_avg, gran_var = powerFeature.extract(theta,
                                            window=32,
                                            overlap=16
                                            )
                # theta = np.expand_dims(theta, axis=1)
                error = e.splitNDArray(theta, 0, 1, True, fs=512, downsample=16)
                no_error = e.splitNDArray(theta, 0, 1, False, fs=512, downsample=16)


                if error_master == []:
                    error_master = error
                    no_error_master = no_error
                else:
                    error_master = np.append(error_master, error, axis=0)
                    no_error_master = np.append(no_error_master, no_error, axis=0)

        np.save(f'{trial_name}_error', error_master)
        np.save(f'{trial_name}_no_error', no_error_master)

    else:
        error_master = np.load(f'{trial_name}_error.npy')
        no_error_master = np.load(f'{trial_name}_no_error.npy')

    #error_master = error_master[..., 1:]
    error_master = np.swapaxes(error_master, 1, 2)

    #no_error_master = no_error_master[..., 1:]
    no_error_master = np.swapaxes(no_error_master, 1, 2)

    error_master *= 10e6  # zscore(error_master)
    no_error_master *= 10e6  # zscore(no_error_master)

    return error_master, no_error_master

error_master_offline, no_error_master_offline = trials('Offline')
error_master_S2, no_error_master_S2 = trials('S2')
error_master_S3, no_error_master_S3 = trials('S3')

offline = np.concatenate([error_master_offline, no_error_master_offline])
truth_offline = np.concatenate([np.ones(error_master_offline.shape[0]), np.zeros(no_error_master_offline.shape[0])])
truth_offline = np.expand_dims(truth_offline, 1)
truth_offline = np.concatenate([truth_offline, 1 - truth_offline], 1)
class_balanace_offline = error_master_offline.shape[0] / no_error_master_offline.shape[0]


S2 = np.concatenate([error_master_S2, no_error_master_S2])
truth_S2 = np.concatenate([np.ones(error_master_S2.shape[0]), np.zeros(no_error_master_S2.shape[0])])
truth_S2 = np.expand_dims(truth_S2, 1)
truth_S2 = np.concatenate([truth_S2, 1 - truth_S2], 1)
class_balanace_S2 = error_master_S2.shape[0] / no_error_master_S2.shape[0]

S3 = np.concatenate([error_master_S3, no_error_master_S3])
truth_S3 = np.concatenate([np.ones(error_master_S3.shape[0]), np.zeros(no_error_master_S3.shape[0])])
truth_S3 = np.expand_dims(truth_S3, 1)
truth_S3 = np.concatenate([truth_S3, 1 - truth_S3], 1)
class_balanace_S3 = error_master_S3.shape[0] / no_error_master_S3.shape[0]

error_master_offline = np.squeeze(error_master_offline)
no_error_master_offline = np.squeeze(no_error_master_offline)
error_master_S2 = np.squeeze(error_master_S2)
no_error_master_S2 = np.squeeze(no_error_master_S2)
error_master_S3 = np.squeeze(error_master_S3)
no_error_master_S3 = np.squeeze(no_error_master_S3)

net = PowerNet()
net.init(error_master_offline.shape[1:])
model = net.compile()

errors = {
    'Offline': {
        'accuracy': []
    },
    'S2': {
        'accuracy': []
    },
    'S3': {
        'accuracy': []
    }
}


class ErrorNoErrorCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        x = model.metrics_names
        errors['Offline']['accuracy'].append(model.evaluate(offline, truth_offline, verbose=2)[1])
        errors['S2']['accuracy'].append(model.evaluate(S2, truth_S2, verbose=2)[1])
        errors['S3']['accuracy'].append(model.evaluate(S3, truth_S3, verbose=2)[1])
        print('Offline Error accuracy: {}, S2 Error accuracy: {}, S3 Error accuracy: {}'.format(
            errors['Offline']['accuracy'][-1], errors['S2']['accuracy'][-1], errors['S3']['accuracy'][-1]))


model.fit(offline, truth_offline,
          batch_size=16,
          epochs=1000,
          verbose=2,
          shuffle=True,
          class_weight={0: 1, 1: class_balanace_offline},
          callbacks=[ErrorNoErrorCallback()],
          )

# model.fit(np.concatenate([offline, S2]), np.concatenate([truth_offline, truth_S2]),
#           batch_size=16,
#           epochs=1000,
#           verbose=2,
#           shuffle=True,
#           class_weight={0: 1, 1: class_balanace_offline},
#           callbacks=[ErrorNoErrorCallback()],
#           )

x = 0
