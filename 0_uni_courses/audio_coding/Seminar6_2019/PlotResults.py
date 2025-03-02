import matplotlib.pyplot as plt
import numpy as np


class PlotResults:

    def __init__(self):

        self.counter = 1

    def plotQIBarkSubbands(self, mTbarkquant):

        plt.figure(self.counter)
        plt.plot(mTbarkquant.T)
        plt.title("The Quantization Indices of the Scalefactors")
        plt.xlabel("The Bark Subbands")
        plt.show()
        self.counter +=1

    def plotQIMDCTSubbands(self, yq):

        plt.figure(self.counter)
        plt.plot(yq[:, 10])
        plt.title("The Quantization Indices of the Subband values")
        plt.xlabel("The MDCT Subbands")
        self.counter += 1

    def plotCompleteSignalData(self, y, mT, y_dq):

        plt.figure(self.counter)
        plt.plot(20 * np.log10(np.abs(y[10, :]) + 1e-2))
        plt.plot(20 * np.log10(mT[10, :] + 1e-2))
        plt.plot(20 * np.log10(np.abs(y_dq[10, :] - y[10, :]) + 1e-2))
        plt.plot(20 * np.log10(np.abs(y_dq[10, :]) + 1e-2))
        plt.legend(('Magnitude Original Signal Spectrum', 'Masking Threshold',
                    'Magnitude Spectrum Reconstructed Signal Error', 'Magnitude Spectrum Reconstructed Signal'))
        plt.xlabel('MDCT subband')
        plt.ylabel("dB")
        self.counter +=1

    def plotOneSubabnd(self, y, y_dq, q_err, mT, subband_index):

        plt.figure(self.counter)
        plt.plot(y[:, subband_index], label='Original Signal')
        plt.plot(y_dq[:, subband_index], label='Reconstructed Signal')
        plt.plot(q_err[:, subband_index], label='Quantization error')
        plt.plot(mT[:, subband_index], label='Masking threshold')
        plt.xlabel('Frequency subband')
        plt.ylabel('Amplitude')
        plt.legend()
        self.counter +=1

    def plotGainandSNR(self, gain, xlabel):
        r"""
        To plot coding gain(size of bin file) and SNR
        param:  gain:
        """


        plt.figure(self.counter)

        binSize = []
        SNR = []
        key_list = []

        for key in gain:

            binSize.append(gain[key]['size'])
            SNR.append(gain[key]['SNR'])
            key_list.append(key)

        index = np.arange(len(binSize))
        bar_width = 0.35

        plt.subplot(2,1,1)
        plt.bar(index, binSize, bar_width, alpha=0.5, label='Gain',color='g')
        plt.xticks(index , (key_list[0], key_list[1], key_list[2]))
        plt.xlabel(xlabel)
        plt.ylabel('size of encoded data')
        plt.legend()

        plt.subplot(2,1,2)
        plt.bar(index, SNR, bar_width, alpha=0.5, label='SNR',color='r')
        plt.xticks(index, (key_list[0], key_list[1], key_list[2]))
        plt.xlabel(xlabel)
        plt.ylabel('SNR')
        plt.legend()

        plt.tight_layout()


    def show(self):

        plt.show()




