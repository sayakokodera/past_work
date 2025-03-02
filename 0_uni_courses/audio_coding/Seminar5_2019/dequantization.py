import pickle
import scipy.io.wavfile as wav
from wav_file_read_play import wavOperations
import numpy as np


class Dequantize:

    def __init__(self):
        self.wave = wavOperations()
        self.reconstructed_data = None

    def binToWaveConvertor(self, index_data = None, stepsize = None):

        if index_data is not None or stepsize is not None:

            self.decode(index_data , stepsize)

        else:
            if index_data is None:
                raise ValueError("index_data is not given")
            if stepsize is None:
                raise ValueError("stepsize is not given")

    def decode(self, index_data = None , stepsize = None):

        if(index_data is not None or stepsize is not None):

            self.reconstructed_data = index_data * stepsize

        else:
            if index_data is None:
                raise ValueError("index_data is not given")
            if stepsize is None:
                raise ValueError("step size is not given")

    def getReconstructedData(self):

        return self.reconstructed_data

