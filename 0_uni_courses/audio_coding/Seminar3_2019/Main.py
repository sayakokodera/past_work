import numpy as np
from PlotResults import PlotResults
from functions import MDCTFilter
from wav_file_read_play import wavOperations
import matplotlib.pyplot as plt1

if __name__=='__main__' :

    N = 512
    L = 2*N
    fs = 44100


    plt = PlotResults()
    wav = wavOperations()

    #Generation of ramp signal

    x = np.arange(fs)/fs

    f= MDCTFilter()


    # Reconstructing a ramp function
    Y = f.analysis_filter(x, N)

    analysis_IR = f.get_analysis_IR()

    ramp_reco, ramp_reco_mat = f.synthesis_filter(Y)
    
    # Reconstructing an audio signal
    sample_rate, wavArrayData = wav.getWavParameters('Track16.wav')

    mono_signal = wavArrayData[:,0]

    mono_signal_trimmed = mono_signal[0:16000]

    mono_signal_filt_downsampled = f.analysis_filter(mono_signal_trimmed, N)

    mono_reco, mono_reco_mat = f.synthesis_filter(mono_signal_filt_downsampled)

    #wav.playWavFile(1,sample_rate, mono_signal_upsampled_filt)

    subband1 = mono_reco_mat[:,0]

    subband2 = mono_reco_mat[:,511]

    # Plots
    plt.plotReconstrRamp(x, ramp_reco)

    plt.plotFirstandLastSubband(subband1, subband2)

    plt.plotReconstrMono(mono_signal_trimmed, mono_reco)

    plt.show()

# =============================================================================
#     plt1.figure(4)
# 
#     plt1.plot(analysis_IR[0,:]) 
#     
#     plt1.plot(analysis_IR[1, :])
# 
#     plt1.plot(analysis_IR[10, :])
# 
#     plt1.plot(analysis_IR[511, :])
# 
#     plt1.show()
# =============================================================================







