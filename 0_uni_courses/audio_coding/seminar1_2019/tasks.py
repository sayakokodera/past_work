from wav_file_read_play import wavOperations
import numpy as np
from plotData import plotData
import decframewk as dequant
from encframewk import Encoder
from ffTransformFile import ffTransform
import os

class tasks:

    def __init__(self):
        self.wav_obj = wavOperations()
        self.plt = plotData()
        self.sample_rate = None
        self.wavArrayData = None
        self.extracted_seconds = None

    def task_1(self):

        file_name = "Track16.wav"

        self.sample_rate, self.wavArrayData = self.wav_obj.getWavParameters(file_name)

        number_of_bits = np.round(np.log2(np.max(abs(self.wavArrayData))))

        print('sample rate :', self.sample_rate, ', number of bits', number_of_bits)

        start_time = 10
        duration = 8

        self.extracted_seconds = self.wav_obj.audioExtract(file_name, start_time, duration)

        #normalized_8_seconds = wav_obj.normalize_fragment(extracted_8_seconds)

        left_channel_data = self.wav_obj.singleChannelData(self.extracted_seconds, 'left')
        right_channel_data = self.wav_obj.singleChannelData(self.extracted_seconds, 'right')

        normalized_left_channel_data = self.wav_obj.normalize_fragment(left_channel_data)
        normalized_right_channel_data = self.wav_obj.normalize_fragment(right_channel_data)

        #self.plt.plotSingleChannel(normalized_left_channel_data, self.sample_rate, 'Left channel data', 'Time',
        #                      'Normalized amplitude')
        #self.plt.plotSingleChannel(normalized_right_channel_data, self.sample_rate, 'Right channel data', 'Time',
        #                      'Normalized amplitude')

        self.plt.plotLeftRightChannel(normalized_left_channel_data, normalized_right_channel_data, self.sample_rate,
                                 'Left and Right channel data', 'Time', 'Normalized amplitude')

        self.wav_obj.playExtractedData(left_channel_data, self.sample_rate, 1)

    def task_2(self):

        waveFileName='Track16.wav'
        start_block = 2

        self.fftobj = ffTransform()
        fftmatrix , freq = self.fftobj.calculateFFT(waveFileName, start_block)
        #fftmatrix=fftobj.calculateFFT.fftMatrix(waveFileName)
        self.plt.plotFFTMatrix(freq, fftmatrix,'Freq in Khz', 'dB', 'ffTOutput')




    def task_3(self):

        fnameinput = 'Track16.wav'
        enc = Encoder()
        enc.read_file(fnameinput)
        enc.normalize_wavdata()
        # 16 bits
        fnamebin_16 = 'encoded.bin'
        enc.encode(bits=16, fnameoutput=fnamebin_16)
        self.wavdata_org= enc.get_wavdata(normalize=True)
        # 8 bits
        fnamebin_8 = 'encoded8bit.bin'
        enc.encode(bits=8, fnameoutput=fnamebin_8)
        self.wavdata_org = enc.get_wavdata(normalize=True)

        # Compare the data size
        print('***Size of each data***')
        print('Original wav data: {}MB'.format(os.stat(fnameinput).st_size / 10 ** 6))
        print('16bits quantized bin data: {}MB'.format(os.stat(fnamebin_16).st_size / 10 ** 6))
        print('8bits quantized bin data: {}MB'.format(os.stat(fnamebin_8).st_size / 10 ** 6))

    def task_4(self):

        dec = dequant.decoder()
        stepSize_16 = 2 / (2 ** 16)
        stepSize_8 = 2 / (2 ** 8)

        #print(self.stepSize_16, self.stepSize_8)

        dec.binToWaveConvertor("encoded.bin", self.sample_rate, stepSize_16 , wavFilename = 'decoded.wav')
        reconstructed_data_16bit = dec.getReconstructedData()

        dec.binToWaveConvertor("encoded8bit.bin", self.sample_rate, stepSize_8, wavFilename= 'decoded8bit.wav' )
        reconstructed_data_8bit = dec.getReconstructedData()

        style ='singlePlot'
        self.plt.plotOriginalReconstructedData(self.wavdata_org, reconstructed_data_16bit, self.sample_rate,
                                               'Original data vs Reconstructed data 16 bits', 'Time', 'Normalized amplitude',
                                                style)
        self.plt.plotOriginalReconstructedData(self.wavdata_org, reconstructed_data_8bit, self.sample_rate,
                                               'Original data vs Reconstructed data 8 bits', 'Time', 'Normalized amplitude',
                                               style)

        #print(reconstructed_data_16bit - reconstructed_data_8bit)

    def showResults(self):

        self.plt.show()

if __name__ == '__main__':

    task = tasks()

    task.task_1()
    task.task_2()
    task.task_3()
    task.task_4()
    task.showResults()


