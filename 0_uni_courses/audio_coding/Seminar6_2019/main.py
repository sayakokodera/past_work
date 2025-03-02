from wav_file_read_play import wavOperations
from MDCT import MDCTFilter
import numpy as np
import scipy.signal as scisig
from psyco import signals
from functions import functions
from PlotResults import PlotResults
import os
from MeasureSNR import MeasureSNR



def parameter_check(parameter_string ,paramter_list):

    measure = MeasureSNR()

    sRate, wavArray = wavOperations().getWavParameters("Audio Samples-20200115/castanets_16.wav")
    DFTResolution = 512
    #barkResolution = 48
    number_of_subbands = DFTResolution / 2

    wavArray = wavArray[:40000]

    # Instanciate the class functions
    f = functions()
    plt = PlotResults()

    #signalToModify = wavArray[:,0]
    signalToModify = wavArray
    AnalysisFilterResult = f.mdct.analysis_filter(signalToModify, number_of_subbands)
    sampleFrequency, t, ys = signals(DFTResolution/2).calculateSTFT(signalToModify)
    ys *= np.sqrt(DFTResolution) / 2 / 0.375  # From book pg125, to match up parsevals energy

    gain = {}
    #quality = 60 #Higher values increase the audio quality, lower values decrease the bit rate.

    for parameter in paramter_list:
        exponent = None
        DFTResolution = 512
        # barkResolution = 48
        number_of_subbands = DFTResolution / 2
        if parameter_string == 'tonalityidx':
            yq, mTbarkquant = f.analysisAndQuantisation(sampleFrequency, ys,AnalysisFilterResult, DFTResolution, tonalityidx = parameter)
        elif parameter_string == 'alpha':
            yq, mTbarkquant = f.analysisAndQuantisation(sampleFrequency, ys, AnalysisFilterResult, DFTResolution,
                                                        alpha=parameter)
        elif parameter_string == 'quality_factor':
            yq, mTbarkquant = f.analysisAndQuantisation(sampleFrequency, ys, AnalysisFilterResult, DFTResolution,
                                                    quality_factor=parameter)
        elif parameter_string == 'non_uniform':
            yq, mTbarkquant = f.analysisAndQuantisation(sampleFrequency, ys, AnalysisFilterResult, DFTResolution,
                                                    non_uniform = parameter)
            exponent = parameter
        elif parameter_string == 'DFTResolution':

            number_of_subbands = parameter / 2

            signalToModify = wavArray
            AnalysisFilterResult = f.mdct.analysis_filter(signalToModify, number_of_subbands)
            sampleFrequency, t, ys = signals(parameter/2).calculateSTFT(signalToModify)
            ys *= np.sqrt(parameter) / 2 / 0.375

            yq, mTbarkquant = f.analysisAndQuantisation(sampleFrequency, ys, AnalysisFilterResult, parameter)


            #print('size of yq',np.size(yq), np.shape(yq))
            print('size of mTbarkquant', np.size(mTbarkquant), np.shape(mTbarkquant))

            print('size of ys', np.shape(ys))

            DFTResolution = parameter



        # Decoding the encoded.bin
        xhat, Xhat = f.decoder('encoded.bin', sampleFrequency,DFTResolution, non_uniform = exponent)
        # quantization error: original data - decoded data
        y_dq = f.get_ydq()
        #y = np.sum(AnalysisFilterResult, axis = 1)
        q_err = AnalysisFilterResult - y_dq
        # Convert q_err into the freq. domain: FFT? STFT?

        SNR = measure.calculateSNR(AnalysisFilterResult, q_err)

        #print(os.stat("encoded.bin").st_size)

        gain[parameter] = {'size' : os.stat("encoded.bin").st_size, 'SNR': SNR}

    # Get the masking threshold
        mT = f.get_maskiing_threshold() # size(41 x 512) is weird... this should be an array, containing only 512 elements

        wavOperations().playExtractedData(xhat, sRate,1)

    subband_index = 30

    plt.plotQIBarkSubbands(mTbarkquant)
    plt.plotQIMDCTSubbands(yq.T)
    plt.plotCompleteSignalData(AnalysisFilterResult, mT, y_dq)
    plt.plotOneSubabnd(AnalysisFilterResult, y_dq, q_err, mT, subband_index)

    plt.plotGainandSNR(gain, parameter_string)

    plt.show()


if __name__ == '__main__':


    tonalityidx_list = [0, 0.5, 1]
    quality_factor_list = [20, 100, 200]
    alpha_list = [0.6, 0.8, 1]
    #non_uniform = ['mu-law']
    DFT_resolution_list = [1024, 512, 256]

    #parameter_check('tonalityidx', tonalityidx_list)
    #parameter_check('non_uniform', non_uniform)
    #parameter_check('quality_factor', quality_factor_list)
    #parameter_check('alpha', alpha_list)
    parameter_check('DFTResolution', DFT_resolution_list)



