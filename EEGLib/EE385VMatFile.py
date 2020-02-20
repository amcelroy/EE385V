from scipy import io
import numpy as np


class EE385VMatFile:
    '''
    Class to make working with the .mat file more modular
    '''

    def __init__(self, filepath):
        self.__mat = {}
        self.__current_action = np.ndarray
        self.__current_goal = np.ndarray
        self.__current_action = np.ndarray
        self.__current_state = np.ndarray
        self.__word_to_write = ''
        self.open(filepath)

    def open(self, filepath):
        '''
        Opens the .mat file from EE385V and attempts to parse it

        :param filepath: Path to .mat file
        :return: Matlab based struct
        '''
        self.__mat = io.loadmat(filepath, mat_dtype=True)['runData']
        self.__current_action = self.__mat['PM'][0][0]['currentAction'][0][0][0]
        self.__current_state = self.__mat['PM'][0][0]['currentState'][0][0]
        self.__current_goal = self.__mat['PM'][0][0]['currentGoal'][0][0]
        self.__word_to_write = self.__mat['PM'][0][0]['word_to_write_ONLINE'][0][0][0][0][0][0]

    def targetWord(self):
        '''
        Get the intended letter for this trial

        :return: Intented Letter
        '''
        return self.__word_to_write

    def goal(self):
        return self.__current_goal

    def actions(self):
        return self.__current_action

    def states(self):
        return self.__current_state


