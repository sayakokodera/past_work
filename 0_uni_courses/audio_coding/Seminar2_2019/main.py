import pyaudio
import matplotlib.pyplot as plt
import signals as si
import plotResults 
import numpy as np

plt.close('all')

if __name__ == '__main__':

    sig = si.signals()
    p = plotResults.plotResults()

    signal_1 = sig.generateSignals(200)
    signal_2 = sig.generateSignals(600)

    time_scale, signal = sig.addTwoSignals(signal_1, signal_2)

    sample_frequency, segment_time, zxx = sig.calculateSTFT(signal)

    magnitude_spectrum = np.sum(np.abs(zxx), axis = 1)/zxx.shape[1] #zxx[:, 1]

    bark_frequency = sig.hz2bark(sample_frequency)

    quantized_Bark = sig.getQuantizedBark(bark_frequency)

    freqbarkMapMat = sig.mappingFreqToBarkDomain(quantized_Bark)

    barkfreqMapMat = sig.mappingfromBarkmat(freqbarkMapMat)

    #mapping DFT bands to the Bark domain

    magnitude_to_bark_domain = sig.mapDFTbandsToBarkbands(np.abs(magnitude_spectrum),freqbarkMapMat)

    mxBark = magnitude_to_bark_domain ** 0.5



    spreading_vector = sig.f_SP_dB()
    spreading_matrix =  sig.spreadingfunctionmat(spreading_vector)

    # Masking threshold
    mTBark = sig.maskingThresholdBark(mxBark, spreading_matrix)#[0, :] = mth for the signal, [1, :] = LTQ
    # Mapping back to the freq. domain
    mTLin = sig.mappingfrombark(mTBark, barkfreqMapMat) #[0, :] = mth for tone, [1, :] = LTQ
    
    ### plotting ###
    p.plotSTFT(segment_time,sample_frequency/(44100/2/1024),zxx)
    # in my opinion, we have to plot the frequecy response
    p.plot_freq_response(magnitude_spectrum)

    p.plotMagnitudeHalfBarkBands(mxBark)

    p.plotFreqToBarkMapMat(freqbarkMapMat)

    p.plotBarkToFreqMapMat(barkfreqMapMat)

    p.plot_spreading_matrix(spreading_matrix)
    
    p.plotMaskingThresholdBark(mTBark)
    p.plotMaskingTInLinDomain(mTLin)


    p.show()
    
    
