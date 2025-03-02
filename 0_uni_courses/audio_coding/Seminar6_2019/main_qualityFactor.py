from wav_file_read_play import wavOperations
from MDCT import MDCTFilter
import numpy as np
import scipy.signal as scisig
from psyco import signals
from functions import functions
from PlotResults import PlotResults
import os
from MeasureSNR import MeasureSNR


measure = MeasureSNR()

sRate, wavArray = wavOperations().getWavParameters("Audio Samples-20200115/speech_16.wav")
DFTResolution = 1024
barkResolution = 48
number_of_subbands = 512

wavArray = wavArray[:40000]

# Instanciate the class functions
f = functions()
plt = PlotResults()

#signalToModify = wavArray[:,0]
signalToModify = wavArray
AnalysisFilterResult = f.mdct.analysis_filter(signalToModify, number_of_subbands)
sampleFrequency, t, ys = signals().calculateSTFT(signalToModify)
ys *= np.sqrt(DFTResolution) / 2 / 0.375  # From book pg125, to match up parsevals energy

print('size of freq and ys', np.size(sampleFrequency), np.size(ys))

quality_factor_list = [20, 100, 200]

gain = {}
#quality = 60 #Higher values increase the audio quality, lower values decrease the bit rate.

for quality_factor in quality_factor_list:

    yq, mTbarkquant = f.analysisAndQuantisation(sampleFrequency, ys,AnalysisFilterResult, DFTResolution, quality_factor = quality_factor)

    # Decoding the encoded.bin
    xhat, Xhat = f.decoder('encoded.bin',sampleFrequency)
    # quantization error: original data - decoded data
    y_dq = f.get_ydq()
    #y = np.sum(AnalysisFilterResult, axis = 1)
    q_err = AnalysisFilterResult - y_dq
    # Convert q_err into the freq. domain: FFT? STFT?

    SNR = measure.calculateSNR(AnalysisFilterResult, q_err)

    print(os.stat("encoded.bin").st_size)

    gain[quality_factor] = {'size' : os.stat("encoded.bin").st_size, 'SNR': SNR}

print(gain)

# Get the masking threshold
mT = f.get_maskiing_threshold() # size(41 x 512) is weird... this should be an array, containing only 512 elements

wavOperations().playExtractedData(xhat, sRate,1)

subband_index = 300

plt.plotQIBarkSubbands(mTbarkquant)
plt.plotQIMDCTSubbands(yq.T)
plt.plotCompleteSignalData(AnalysisFilterResult, mT, y_dq)
plt.plotOneSubabnd(AnalysisFilterResult, y_dq, q_err, mT, subband_index)

plt.plotGainandSNR(gain, 'quality factor')

plt.show()