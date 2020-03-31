import matplotlib

from Feature.BCIFeature import BCIFeature
from stats import Stats

matplotlib.use("TkAgg")

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

root = parser['DATA']['root']

dsl = DatasetLoader(root)

offline_dict = dsl.getAll()

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
                                                                            frequency_range=[4, 9])
        return feature_vol, grand_avg, grand_var, freq, time
    else:
        feature_vol, grand_avg, grand_var = feature.extract(error, window, overlap)

    return feature_vol, grand_avg, grand_var

subject_list = []
for subject in offline_dict.keys():
    subject_list.append(subject)

trial_list = ['Offline', 'S2', 'S3']

for trial in trial_list:

    topoplot_data = []

    for subject in subject_list:

        runs = offline_dict[subject][trial]

        grand_avg_fig, grand_avg_axis = plt.subplots(len(channel_names_spellers), 3)

        count = 0
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
            )
            no_error = e.getEEGTrials(pretime=.5, posttime=1, error=False, plot=False, gdf=theta)

            error = e.CARFilter(error)
            no_error = e.CARFilter(no_error)

            stftfeature = STFTFeature()
            feature_vol, grand_avg, grand_var, freq, time = applyFeature(error, stftfeature, 512, 496)
            zero_time = np.argwhere(time == 0)
            zero_time = zero_time[0][0]
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
        #grand_grand_avg_error = np.log10(grand_grand_avg_error)
        grand_grand_avg_error = (grand_grand_avg_error - grand_grand_avg_error.min()) / (
                grand_grand_avg_error.max() - grand_grand_avg_error.min())
        grand_grand_avg_error = np.mean(grand_grand_avg_error, axis=0)

        grand_grand_avg_no_error = np.array(grand_avg_no_error_array)
        #grand_grand_avg_no_error = np.log10(grand_grand_avg_no_error)
        grand_grand_avg_no_error = (grand_grand_avg_no_error - grand_grand_avg_no_error.min()) / (
                grand_grand_avg_no_error.max() - grand_grand_avg_no_error.min())
        grand_grand_avg_no_error = np.mean(grand_grand_avg_no_error, axis=0)

        grand_grand_var_error = np.array(grand_var_error_array)
        #grand_grand_var_error = np.log10(grand_grand_var_error)
        grand_grand_var_error = (grand_grand_var_error - grand_grand_var_error.min()) / (
                grand_grand_var_error.max() - grand_grand_var_error.min())
        grand_grand_var_error = np.mean(grand_grand_var_error, axis=0)

        grand_grand_var_no_error = np.array(grand_var_no_error_array)
        # grand_grand_var_no_error = np.log10(grand_grand_var_no_error)
        grand_grand_var_no_error = (grand_grand_var_no_error - grand_grand_var_no_error.min()) / (
                grand_grand_var_no_error.max() - grand_grand_var_no_error.min())
        grand_grand_var_no_error = np.mean(grand_grand_var_no_error, axis=0)

        stftfeature.plot(grand_avg_axis[..., 0], grand_grand_avg_error, zero_time=zero_time)
        stftfeature.plot(grand_avg_axis[..., 1], grand_grand_avg_no_error, zero_time=zero_time)
        stftfeature.addYLabels(grand_avg_axis[..., 0], e.getChannelNames())
        stftfeature.addXLabels(grand_avg_axis[..., 0], time=time)
        stftfeature.addXLabels(grand_avg_axis[..., 1], time=time)
        stftfeature.addTitle(grand_avg_axis[..., 0], title='Normalized STFT of Error EEG')
        stftfeature.addTitle(grand_avg_axis[..., 1], title='Normalized STFT of No Error EEG')

        corr = []
        c = Stats()
        for x in range(grand_grand_avg_error.shape[0]):
            t = c.squared_error(grand_grand_avg_error[x],
                                    grand_grand_avg_no_error[x],
                                    )
            corr.append(t)
        corr = np.array(corr)
        stftfeature.plot(grand_avg_axis[..., 2], corr, zero_time=zero_time)
        stftfeature.addXLabels(grand_avg_axis[..., 2], time=time)
        stftfeature.addFreqLabels(grand_avg_axis[..., 2], tick_labels=[4, 5, 6, 7, 8])
        stftfeature.addTitle(grand_avg_axis[..., 2], title='MSE of Error / No Error')
        grand_avg_fig.suptitle('Grand Average for Error, No Error, MSE - {}, Trial - {}'.format(subject, trial))

        # average all freq. bands
        corr = np.mean(corr, 1)
        corr = (corr - corr.min()) / (
                corr.max() - corr.min())
        corr -= .5
        topoplot_data.append(corr)

        # plt.show()
        grand_avg_fig.set_size_inches((25, 18), forward=False)
        plt.savefig('grand_avg_{}_{}.png'.format(trial, subject), dpi=100)

        # fig_corr, ax_corr = plt.subplots(len(channel_names_spellers), len(offline_dict.keys()) - 1)
        # for x in range(0, len(grand_avg_error_array) - 1):
        #     for y in range(grand_avg_error_array[x].shape[0]):
        #         covar = np.square(grand_avg_error_array[x][y, ...] - grand_avg_error_array[x + 1][y, ...])
        #         covar /= grand_var_error_array[x][y, ...]**2
        #         ax_corr[y, x].imshow(covar)
        # fig_corr.show()

    topoplot_data = np.array(topoplot_data)
    fig = e.topoplot(topoplot_data, times=time)
    fig.set_size_inches((25, 18), forward=False)
    plt.savefig('topoplot_{}_all.png'.format(trial), dpi=100)

x = 0
