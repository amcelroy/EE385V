import mne
import scipy
from scipy import io
from mne import Epochs
from mne.channels import Layout
import os
from pathlib import Path

from EEGLib.EE385VMatFile import EE385VMatFile


class EEG:
    def __init__(self):
        self.__gdf = []
        self.__mat = []

    def open(self, filepath):
        '''
        @param filepath: Path to GDF or .MAT file
        @:return Tuple (gdf, EE385VMatFile)
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
        mat = EE385VMatFile(mat_path)
        self.__gdf = gdf
        self.__mat = mat
        return (gdf, mat)

    def getRawEEG(self, channels=['']):
        '''
        Packages https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data

        :param channels: String list of EEG channels to fetch
        :return: Tuple of ([EEG Channel, Data], Time)
        '''
        if channels == ['']:
            return gdf.get_data(return_times=True)
        else:
            return gdf.get_data(picks=channels, return_times=True)


if __name__ == "__main__":
    e = EEG()
    (gdf, mat) = e.open(
        '/home/amcelroy/Code/EE385V/BCI Course 2020/ErrPSpeller/Subject1/Offline/ad4_raser_offline_offline_171110_172431.gdf')

    channels = e.getRawEEG()
    mat.targetLetter()
    mat.ringBuffer()
    print(gdf.info)
