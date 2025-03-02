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

        self.max = None
        self.delta_max = None

    def analysisAndQuantisation(self, sampleFrequency, ys, AnalysisFilterResult, DFTResolution, alpha = None,
                                quality_factor = 100, non_uniform = None, tonalityidx = None):
        signal = signals(DFTResolution/2)

        barkfreq = signal.hz2bark(sampleFrequency)
        quantized_bark = signal.getQuantizedBark(barkfreq)

        W = signal.mappingFreqToBarkDomain(quantized_bark)
        W_inv = signal.mappingfrombarkmat(W, DFTResolution)

        spreading_vector_db = signal.f_SP_dB()
        #spreading_matrix = signal.spreadingfunctionmat(spreading_vector_db, alpha)
        spreading_matrix = signal.spreadingfunctionmat(spreading_vector_db, tonalityidx = tonalityidx)

        # magnitude_spectrum = np.sum(np.abs(ys), axis=1) / ys.shape[1]
        ysabs = np.abs(ys)
        ysabs = ysabs.T

        magMappedBark = signal.mapDFTbandsToBarkbands(np.abs(ysabs), W)

        mxBark = magMappedBark ** 0.5

        mTBark = signal.maskingThresholdBark(mxBark, spreading_matrix, alpha)/(quality_factor/100)

        mTbarkquant = np.round(np.log2(mTBark) * 4)
        mTbarkquant = np.clip(mTbarkquant, 0, None)
        mTBarkDeQuantized = np.power(2, mTbarkquant / 4)

        self.mT = signal.mappingfrombark(mTBarkDeQuantized, W_inv)
        #print('Masking threshold with the size {}'.format(self.mT.shape))

        delta = self.mT * 2
        delta = delta[:-1, :]  # leaving the last data to match dimension with output of the MDCT analysis Filterbank

        if non_uniform is not None:

            if non_uniform == 'mu-law':

                # Non-uniform Quantisation
                self.max = np.abs(AnalysisFilterResult).max()
                self.delta_max = np.abs(delta).max()
                delta = delta / self.delta_max
                # mu-law companding
                AnalysisFilterResult = AnalysisFilterResult / (self.max)
                AnalysisFilterResult = np.sign(AnalysisFilterResult) * (np.log(1 + 255 * np.abs(AnalysisFilterResult))) / (np.log(256))


        yq = np.round((AnalysisFilterResult) / delta).astype(np.int8)

        # pickle.dump((yq, mTBarkQuantized), open("encoded.bin","wb"))
        pickle.dump((yq, mTbarkquant), open("encoded.bin", "wb"))

        return yq, mTbarkquant

    def decoder(self, fname,sampleFrequency, DFTResolution, non_uniform = None):
        """
        (1) Load the file
        (2) Convert the data from binary to "normal" according to the bit size
        (3) Dequantize the data
        (4) Synthesis filter
        """

        signal = signals(DFTResolution / 2)
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

            self.ydeq = (yq * delta)

            if non_uniform is not None:

                if non_uniform == 'mu-law':

                    self.ydeq = self.ydeq / self.delta_max

                    self.ydeq = np.sign(self.ydeq) * ((256 ** np.abs(self.ydeq)) - 1) / 255

                    self.ydeq = self.ydeq * self.max


            # Synthesis filter
            xhat, Xhat = self.mdct.synthesis_filter(self.ydeq)
            return xhat, Xhat
    
    
    def get_maskiing_threshold(self):

        return self.mT

    def get_ydq(self):
        #return np.sum(self.ydeq, axis=1)
        return self.ydeq

