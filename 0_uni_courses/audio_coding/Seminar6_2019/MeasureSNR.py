import numpy as np

class MeasureSNR:

    def calculateSNR(self, originalSignal, error):

        signal_power = np.sum(np.power(originalSignal, 2))
        error_power = np.sum(np.power(error, 2))


        SNR = (signal_power / error_power)

        print('SNR', SNR)

        SNR_in_db = 10 * np.log10(SNR)

        print('dB',SNR_in_db)

        return SNR_in_db
