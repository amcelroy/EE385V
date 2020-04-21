
import os
from configparser import ConfigParser
import numpy as np
from keras.callbacks import Callback
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

# Get dictionaries with filenames for each dataset
offline_dict = dsl.getTrials()[0]
S2_dict = dsl.getTrials()[1]
S3_dict = dsl.getTrials()[2]

channels_of_interest = ['FCz', 'CZ', 'FC1', 'FC2', 'CPz', 'C4', 'CP4']

# adding each trial with its own variable for error_master, no_error_master 
def trials (trial):
    error_master = []
    no_error_master = []
    trial_dicts = [offline_dict, S2_dict, S3_dict]
    # Create or load the data
    if not (os.path.exists(f'{trial}_error.npy') and os.path.exists(f'{trial}_no_error.npy')):
        for trial_num in trial_dicts:
            for subject in trial_num.keys():
                runs = trial_num[subject]
#                count = 0
                for run in runs:
                    path = os.path.join(root, subject, trial, run)
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
        
            np.save(f'{trial}_error', error_master)
            np.save(f'{trial}_no_error', no_error_master)
    else:
        error_master = np.load(f'{trial}_error.npy')
        no_error_master = np.load(f'{trial}_no_error.npy')
    
    error_master = error_master[..., 1:]
    error_master = np.swapaxes(error_master, 2, 3)
    error_master = np.swapaxes(error_master, 1, 3)
    
    no_error_master = no_error_master[..., 1:]
    no_error_master = np.swapaxes(no_error_master, 2, 3)
    no_error_master = np.swapaxes(no_error_master, 1, 3)
    
    error_master *= 10e6 #zscore(error_master)
    no_error_master *= 10e6 #zscore(no_error_master)
    
    return error_master, no_error_master 


error_master_offline = trials('Offline')[0]
no_error_master_offline = trials('Offline')[1]

error_master_S2 = trials('S2')[0]
no_error_master_S2 = trials('S2')[1]

#error_master_S3 = trials('S3')[0]
#no_error_master_S3 = trials('S3')[1]

# Split the error data into 80/20 test/train
indecies = np.arange(error_master_offline.shape[0])
np.random.shuffle(indecies)

validation_index = indecies[0:no_error_master_offline.shape[0]]
train_index = indecies[no_error_master_offline.shape[0]:]

train_set = error_master_offline[train_index, ...]
val_set = error_master_offline[validation_index, ...]

snet = stftNet.STFTNet()
snet.init(train_set.shape[1:])
model = snet.compile()

class ErrorNoErrorCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        x = model.evaluate(no_error_master, no_error_master, verbose=2)
        y = model.evaluate(val_set, val_set, verbose=2)
        print('No Error Loss: {}, Error Loss: {}'.format(x, y))
        print('Percent Diff: {}'.format(100*(abs(y - x) / y)))


model.fit(train_set, train_set,
          batch_size=128,
          epochs=1000,
          verbose=2,
          shuffle=True,
          callbacks=[ErrorNoErrorCallback()],
          )

x = 0