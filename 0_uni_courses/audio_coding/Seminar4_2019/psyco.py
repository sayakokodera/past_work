import numpy as np
import scipy.signal as sc
import demo_moodle.psyacmodel as ps

class signals():


    def __init__(self):

        self.alpha = 0.8
        self.sampling_freq = 44100
        self.duration = 3 * 60
        self.dft_resolution = 512
        self.bark_resolution = 48

    def generateSignals(self, sub_band, amplitude):
        fmax = self.sampling_freq/2 # maximal freq. component (i.e. f_Nyquist)
        f0 = fmax / self.dft_resolution # sampling interval for f
        t = np.arange(3*60*self.sampling_freq)/self.sampling_freq
        
        signal = np.sin(2*np.pi*sub_band*f0*t)* amplitude        
        #signal = np.sin(2 * np.pi /(2 * self.dft_resolution) * sub_band * np.arange(self.sampling_freq * self.duration)) * 1000

        return  signal

    def addTwoSignals(self, signal1,signal2):

        resultant = signal1 + signal2

        return np.linspace(0, self.duration, len(resultant)), resultant

    def calculateSTFT(self, signal):

        sample_frequency, segment_time, zxx = sc.stft(signal, fs = 2 * np.pi, nperseg= 2 * self.dft_resolution -1)

        return sample_frequency, segment_time, zxx

    def hz2bark(self, f):

        return ps.hz2bark(f)

    def bark2hz(self, Brk):

        return  ps.bark2hz(Brk)


    def getQuantizedBark(self, barkFrequency):

         max_bark= barkFrequency.max()
         step_size= max_bark/(self.bark_resolution-1)
         return np.round(barkFrequency/step_size)

    def mappingFreqToBarkDomain(self, quantized_Bark):

        barkFreqMapMat = np.zeros((self.bark_resolution, self.dft_resolution))

        for i in range(self.bark_resolution):
            barkFreqMapMat[i, :] = (quantized_Bark == i)

        return barkFreqMapMat

    def mappingfromBarkmat(self, W):

        return ps.mappingfrombarkmat(W, 2*self.dft_resolution)

    def f_SP_dB(self):

        return ps.f_SP_dB(self.sampling_freq/2, self.bark_resolution)

    def spreadingfunctionmat(self,spreading_vector):

        return ps.spreadingfunctionmat(spreading_vector,self.alpha,self.bark_resolution)

    def mapDFTbandsToBarkbands(self, magnitude_spectrum, mapping_matrix) :
        psd = magnitude_spectrum**2
        return np.dot(psd, mapping_matrix.T) # within This is adding the power of the subbands, vector of 1x48


    def mapping2bark(self, toneSTFT, W, Nfft):

        return ps.mapping2bark(toneSTFT, W, Nfft)

    def maskingThresholdBark(self, mXbark,spreadingfuncmatrix):
        """ 
        Retrun
        ------
            masking threshold : np.ndarray(2, bark_resolution)
                1st row = masking threshold of the signal
                2nd row = LTQ
        """
        mTbark = np.dot(mXbark ** self.alpha, spreadingfuncmatrix ** self.alpha)
        mTbark = mTbark ** (1.0 / self.alpha)
        return mTbark

        #return ps.maskingThresholdBark(mXbark,spreadingfuncmatrix, self.alpha, self.sampling_freq,self.bark_resolution)

    def mappingfrombark(self, mthBark, W_inv):
        
        return ps.mappingfrombark(mthBark, W_inv, 2*self.dft_resolution)

    def mapping2barkmat(self, fS, Nbark, Nfft):

        return  ps.mapping2barkmat(fS, Nbark, Nfft)

    def mappingfrombarkmat(self, W, Nfft):

        return ps.mappingfrombarkmat(W, Nfft)







