from wav_file_read_play import wavOperations
from MDCT import MDCTFilter
import numpy as np
import scipy.signal as scisig
from psyco import signals
import pickle
from MDCT import MDCTFilter

class functions():

    def __init__(self):
        self.mT = None
        self.mdct = MDCTFilter()

    def analysisAndQuantisation(self, sampleFrequency, ys, AnalysisFilterResult, DFTResolution, barkResolution,
                                maxFreq):
        signal = signals()

        barkfreq = signal.hz2bark(sampleFrequency)
        quantized_bark = signal.getQuantizedBark(barkfreq)

        W = signal.mappingFreqToBarkDomain(quantized_bark)
        W_inv = signal.mappingfrombarkmat(W, DFTResolution)

        spreading_vector_db = signal.f_SP_dB()
        spreading_matrix = signal.spreadingfunctionmat(spreading_vector_db)

        # magnitude_spectrum = np.sum(np.abs(ys), axis=1) / ys.shape[1]
        ysabs = np.abs(ys)
        ysabs = ysabs.T

        magMappedBark = signal.mapDFTbandsToBarkbands(np.abs(ysabs), W)

        mxBark = magMappedBark ** 0.5

        mTBark = signal.maskingThresholdBark(mxBark, spreading_matrix)

        mTbarkquant = np.round(np.log2(mTBark) * 4)
        mTbarkquant = np.clip(mTbarkquant, 0, None)
        mTBarkDeQuantized = np.power(2, mTbarkquant / 4)

        self.mT = signal.mappingfrombark(mTBarkDeQuantized, W_inv)
        print('Masking threshold with the size {}'.format(self.mT.shape))

        delta = self.mT * 2
        delta = delta[:-1, :]  # leaving the last data to match dimension with output of the MDCT analysis Filterbank

        yq = np.round(AnalysisFilterResult / delta)

        # pickle.dump((yq, mTBarkQuantized), open("encoded.bin","wb"))
        pickle.dump((yq, mTbarkquant), open("encoded.bin", "wb"))

        return yq, mTbarkquant

        #print(delta, np.shape(delta),np.shape(AnalysisFilterResult))
    def decoder(self, fname,sampleFrequency):
        """
        (1) Load the file
        (2) Convert the data from binary to "normal" according to the bit size
        (3) Dequantize the data
        (4) Synthesis filter
        """

        signal = signals()
        # load the file
        if fname == None:
            raise AttributeError('The file name should be specified to decode.')
        else:
            y_bin = open(fname, "rb")  # binary data
            # Convert the data from binary to
            yq, mTbarkquant = pickle.load(y_bin)  # stepsize should be an array of 512, not a matrix
            mTbarkdequant = np.power(2, mTbarkquant / 4)
            barkfreq = signal.hz2bark(sampleFrequency)
            quantized_bark = signal.getQuantizedBark(barkfreq)
            W = signal.mappingFreqToBarkDomain(quantized_bark)
            W_inv = signal.mappingfrombarkmat(W, 1024)

            mT = signal.mappingfrombark(mTbarkdequant, W_inv)
            # Dequantization
            delta = mT * 2
            delta = delta[:-1,:]
            self.ydeq = yq * delta
            #self.y_dq = y_q * stepsize  # in freq./z-domain
            # Synthesis filter
            xhat, Xhat = self.mdct.synthesis_filter(self.ydeq)
            return xhat, Xhat
    
    
    def get_maskiing_threshold(self):

        return self.mT

    def get_ydq(self):
        #return np.sum(self.ydeq, axis=1)
        return self.ydeq

