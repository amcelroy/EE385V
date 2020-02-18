from scipy import io


class EE385VMatFile:
    '''
    Class to make working with the .mat file more modular
    '''

    def __init__(self, filepath):
        self.__mat = {}
        self.open(filepath)

    def open(self, filepath):
        '''
        Opens the .mat file from EE385V and attempts to parse it

        :param filepath: Path to .mat file
        :return: Matlab based struct
        '''
        self.__mat = io.loadmat(filepath, mat_dtype=True)['runData']
        return self.__mat

    def targetLetter(self):
        '''
        Get the intended letter for this trial

        :return: Intented Letter
        '''
        letter = self.__mat['targetLetter'][0][0][0]
        letter = str(letter)
        return letter

    def ringBuffer(self):
        '''
        Fetches the ringBuffer for this trial

        :return:
        '''
        buffer = self.__mat['ringbuffer'][0][0]
        return buffer
