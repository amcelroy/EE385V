import matplotlib
from scipy.signal import correlate2d

from Feature.BCIFeature import BCIFeature
from Feature.PowerFeature import PowerFeature
from Stats.correlation import Correlation

matplotlib.use("TkAgg")

import os

import scipy

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

grand_avg_error_array = []
grand_avg_no_error_array = []
grand_var_error_array = []
grand_var_no_error_array = []


def applyFeature(triggerSplitVolume=np.ndarray, feature=BCIFeature, window=64, overlap=48):
    grand_grand_avg_error = []
    grand_grand_avg_var = []

    if isinstance(feature, STFTFeature):
        feature_vol, grand_avg, grand_var, freq, time = stftfeature.extract(triggerSplitVolume,
                                                                            window=window,
                                                                            overlap=overlap,
                                                                            pre_trigger_time=.5,
                                                                            fs=512,
                                                                            frequency_range=[4, 8])
        return feature_vol, grand_avg, grand_var, freq, time
    else:
        feature_vol, grand_avg, grand_var = feature.extract(error, window, overlap)

    return feature_vol, grand_avg, grand_var


for subject in offline_dict.keys():
    runs = offline_dict[subject]

    grand_avg_fig, grand_avg_axis = plt.subplots(len(channel_names_spellers), 3)

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
        )
        no_error = e.getEEGTrials(pretime=.5, posttime=1, error=False, plot=False, gdf=theta)

        error = e.CARFilter(error)
        no_error = e.CARFilter(no_error)

        stftfeature = STFTFeature()
        feature_vol, grand_avg, grand_var, freq, time = applyFeature(error, stftfeature, 512, 496)
        grand_avg_error_array.append(grand_avg)
        grand_var_error_array.append(grand_var)

        # powerfeature = PowerFeature()
        # feature_vol, grand_avg, grand_var = applyFeature(error, stftfeature, 48, 32)

        feature_vol, grand_avg, grand_var, freq, time = applyFeature(no_error, stftfeature, 512, 496)
        grand_avg_no_error_array.append(grand_avg)
        grand_var_no_error_array.append(grand_var)
        count += 1

    # plt.show()
    # fig_error.show()
    # fig_no_error.show()

    grand_grand_avg_error = np.array(grand_avg_error_array)
    grand_grand_avg_error = np.log10(grand_grand_avg_error)
    grand_grand_avg_error = (grand_grand_avg_error - grand_grand_avg_error.min()) / (
            grand_grand_avg_error.max() - grand_grand_avg_error.min())
    grand_grand_avg_error = np.mean(grand_grand_avg_error, axis=0)

    grand_grand_avg_no_error = np.array(grand_avg_no_error_array)
    grand_grand_avg_no_error = np.log10(grand_grand_avg_no_error)
    grand_grand_avg_no_error = (grand_grand_avg_no_error - grand_grand_avg_no_error.min()) / (
            grand_grand_avg_no_error.max() - grand_grand_avg_no_error.min())
    grand_grand_avg_no_error = np.mean(grand_grand_avg_no_error, axis=0)

    grand_grand_var_error = np.array(grand_var_error_array)
    grand_grand_var_error = np.log10(grand_grand_var_error)
    grand_grand_var_error = (grand_grand_var_error - grand_grand_var_error.min()) / (
            grand_grand_var_error.max() - grand_grand_var_error.min())
    grand_grand_var_error = np.mean(grand_grand_var_error, axis=0)

    grand_grand_var_no_error = np.array(grand_var_no_error_array)
    grand_grand_var_no_error = np.log10(grand_grand_var_no_error)
    grand_grand_var_no_error = (grand_grand_var_no_error - grand_grand_var_no_error.min()) / (
            grand_grand_var_no_error.max() - grand_grand_var_no_error.min())
    grand_grand_var_no_error = np.mean(grand_grand_var_no_error, axis=0)

    stftfeature.plot(grand_avg_axis[..., 0], grand_grand_avg_error)
    stftfeature.plot(grand_avg_axis[..., 1], grand_grand_avg_no_error)
    stftfeature.addYLabels(grand_avg_axis[..., 0], e.getChannelNames())
    stftfeature.addXLabels(grand_avg_axis[..., 0], time=time)
    stftfeature.addXLabels(grand_avg_axis[..., 1], time=time)
    stftfeature.addTitle(grand_avg_axis[..., 0], title='STFT of Error EEG')
    stftfeature.addTitle(grand_avg_axis[..., 1], title='STFT of No Error EEG')

    corr = []
    c = Correlation()
    for x in range(grand_grand_avg_error.shape[0]):
        t = c.cross_correlation(grand_grand_avg_error[x],
                                grand_grand_avg_no_error[x],
                                grand_grand_var_error[x],
                                grand_grand_var_no_error[x])
        corr.append(t)
    corr = np.array(corr)
    stftfeature.plot(grand_avg_axis[..., 2], corr)
    stftfeature.addXLabels(grand_avg_axis[..., 2], time=time)
    stftfeature.addTitle(grand_avg_axis[..., 2], title='Cross Correlation of Error / No Error')

    plt.show()

    # fig_corr, ax_corr = plt.subplots(len(channel_names_spellers), len(offline_dict.keys()) - 1)
    # for x in range(0, len(grand_avg_error_array) - 1):
    #     for y in range(grand_avg_error_array[x].shape[0]):
    #         covar = np.square(grand_avg_error_array[x][y, ...] - grand_avg_error_array[x + 1][y, ...])
    #         covar /= grand_var_error_array[x][y, ...]**2
    #         ax_corr[y, x].imshow(covar)
    # fig_corr.show()

    x = 0
