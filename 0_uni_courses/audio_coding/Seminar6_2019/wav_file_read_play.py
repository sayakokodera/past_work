import scipy.io.wavfile as wav
import pyaudio
import numpy as np

class wavOperations:

    def getWavParameters(self, wavFileName):

        sample_rate, wavArrayData = wav.read(wavFileName)

        return sample_rate, wavArrayData

    def getChannelCount(self, wavData):

        return wavData.shape[1]

    def playWavFile(self, wavFileName):

        pAudio = pyaudio.PyAudio()

        sample_rate, wavArrayData = self.getWavParameters(wavFileName)

        channels = self.getChannelCount(wavArrayData)

        stream = pAudio.open(format = pyaudio.paInt16, channels = channels,
                        rate = sample_rate, output = True)

        stream.write(wavArrayData.astype(np.int16).tostring())

        stream.stop_stream()
        stream.close()
        pAudio.terminate()

    def playExtractedData(self, wavArrayData, sample_rate, channels):

        pAudio = pyaudio.PyAudio()

        stream = pAudio.open(format = pyaudio.paInt16, channels = channels,
                        rate = sample_rate, output = True)

        stream.write(wavArrayData.astype(np.int16).tostring())

        stream.stop_stream()
        stream.close()
        pAudio.terminate()

    def audioExtract(self, wavFilename, start_time, interval):

        sample_rate, wavArrayData = self.getWavParameters(wavFilename)

        start_sample = start_time * sample_rate

        interval_samples = interval * sample_rate

        audio_extracted = wavArrayData[start_sample:start_sample+interval_samples]

        return  audio_extracted

    def normalize_fragment(self, wavArrayData):

        noramlize_data = wavArrayData / np.int16(np.max(abs(wavArrayData)))

        return noramlize_data

    def singleChannelData(self, wavArrayData, channel_index):

        if channel_index == 'left':
            return np.ascontiguousarray(wavArrayData[:,0], np.int16)
        elif channel_index == 'right':
            return np.ascontiguousarray(wavArrayData[:,1], np.int16)
        else:
            raise ValueError('Channel index is not type')

    def calculateStepSize(self, waveArrayData, number_of_bits):
        max = np.max(waveArrayData)
        min = np.min(waveArrayData)
        amplitudeDifference = int(max) - int(min)
        steps = 2 ** number_of_bits
        stepsize = amplitudeDifference / steps
        return stepsize

    def calculateNormalizedStepSize(self, number_of_bits):
        max = 1
        min = -1
        amplitudeDifference = max - min
        steps = 2 ** number_of_bits
        stepsize = amplitudeDifference / steps
        return stepsize

















