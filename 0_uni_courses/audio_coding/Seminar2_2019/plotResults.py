import numpy as np
import matplotlib.pyplot as plt

class plotResults():

    def __init__(self):

        self.counter = 1
        pass

    def plotSTFT(self, time, freq, mag):

        plt.figure(self.counter)
        plt.pcolormesh(time, freq, np.abs(mag))
        plt.title('Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Number of sub-band')
        self.counter += 1
        
    def plot_freq_response(self, spectrum):

        plt.figure(self.counter)
        plt.plot(np.abs(spectrum))
        plt.title('Frequency response of the signal')
        plt.xlabel('Freq. subband')
        plt.ylabel('Specrum')
        self.counter += 1

    def plot_spreading_matrix(self, spreading_matrix):

        plt.figure(self.counter)
        plt.imshow(spreading_matrix)
        plt.title('Spreading matrix')
        plt.xlabel('Bark subbands')
        plt.ylabel('Bark subbands')
        self.counter += 1
        

    def plotFreqToBarkMapMat(self,FreqBarkMapMat):

        plt.figure(self.counter)
        plt.imshow(FreqBarkMapMat[:,:600])
        plt.title('Matrix for Uniform to Bark Mapping')
        plt.xlabel('Uniform Subbands')
        plt.ylabel('Bark Subbands')
        self.counter += 1

    def plotBarkToFreqMapMat(self, barkToFreqMapMat):

        plt.figure(self.counter)
        plt.imshow(barkToFreqMapMat[:600, :])
        plt.title('Matrix for Bark to Uniform Mapping')
        plt.xlabel('Bark Subbands')
        plt.ylabel('Uniform Subbands')
        self.counter += 1


    def plotMagnitudeHalfBarkBands(self, BarkFrequency):

        plt.figure(self.counter)
        plt.plot(BarkFrequency)
        plt.title("Magnitude Spectrum mapped to 1/2 bark")
        plt.ylabel('Amplitude in dB')
        plt.xlabel('Frequency in Bark')
        self.counter +=1

    def plotMaskingThresholdBark(self, maskingThresholdBarkDomain):

        plt.figure(self.counter)
        plt.plot(20*np.log10(maskingThresholdBarkDomain[0, :] + 10**(-6)), label = 'excl. LTQ')
        plt.plot(20*np.log10(np.maximum(maskingThresholdBarkDomain[0, :], maskingThresholdBarkDomain[1, :]) + 10**(-6)), 
                 label = 'incl. LTQ')
        plt.title('Masking Threshold in Bark Domain')
        plt.legend()
        plt.xlabel('Bark subband')
        plt.ylabel('[dB]')
        self.counter +=1

    def plotMaskingTInLinDomain(self, maskingLinearDomain):

        plt.figure(self.counter)
        plt.plot(20*np.log10(maskingLinearDomain[0, :] + 10**(-6)), label = 'excl. LTQ')
        plt.plot(20*np.log10(np.maximum(maskingLinearDomain[0, :], maskingLinearDomain[1, :]) + 10**(-6)), 
                 label = 'incl. LTQ')
        plt.title('Masking Threshold in Freq. Domain')
        plt.legend()
        plt.xlabel('Freq. subband')
        plt.ylabel('[dB]')
        self.counter += 1

    def show(self):

        plt.show()

