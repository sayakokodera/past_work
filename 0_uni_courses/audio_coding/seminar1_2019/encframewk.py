# -*- coding: utf-8 -*-
"""
Audio Coding: Seminar 1 

"""

import pickle
from wav_file_read_play import wavOperations
import numpy as np

class Encoder:
    def __init__(self, wavdata = None, samplingrate = None):
        self.wave = wavOperations()
        self.wavdata = np.array(wavdata)
        self.samplingrate = samplingrate
        self.datanormed = None
        self.stepsize = None
        #self.max = None

    def read_file(self, fnameinput): 
        self.samplingrate, self.wavdata = self.wave.getWavParameters(fnameinput)
        
    def normalize_wavdata(self):
        self.datanormed = self.wavdata / abs(self.wavdata).max()
        #self.max = abs(self.wavdata).max() -> we do not need this, because the max is now 1

    def encode(self, bits, fnameoutput): 
        if(bits != None):
            if(self.samplingrate == None):
                self.readFile()
            self.normalize_wavdata()
            self.stepsize = self.wave.calculateNormalizedStepSize(bits)
            q_idx = np.round(self.datanormed / self.stepsize)
            if(bits == 8):
                #q_idx = q_idx+128
                q_idx = q_idx.astype(np.int8)
                pickle.dump(q_idx, open(fnameoutput, "wb"), pickle.HIGHEST_PROTOCOL)
            elif(bits == 16):
                q_idx = q_idx.astype(np.int16)
                pickle.dump(q_idx, open(fnameoutput, "wb"), pickle.HIGHEST_PROTOCOL)
            else:
                print("No action can be performed")
                return
            
    def get_wavdata(self, normalize = False):
        if normalize == True:
            return self.datanormed
        else:
            return self.wavdata
        
