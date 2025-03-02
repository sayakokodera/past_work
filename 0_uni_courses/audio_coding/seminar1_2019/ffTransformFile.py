import numpy as np
import wav_file_read_play


class ffTransform:

    def __init__(self):

        self.wavObj = wav_file_read_play.wavOperations()

    def calculateFFT(self, waveFileName, start_block):



        sample_rate, wavArrayData = self.wavObj.getWavParameters(waveFileName)

        blockSize = 1024

        number_of_blocks = 4

        WavArrayLeftCh = np.ascontiguousarray(wavArrayData[:, 0], np.int16)
        WavArrayRightCh = np.ascontiguousarray(wavArrayData[:, 1], np.int16)

        block_4096 = WavArrayRightCh[start_block * blockSize :blockSize * number_of_blocks + start_block * blockSize]

        block_4096_matrix = block_4096.reshape(number_of_blocks, blockSize)

        time_steps = 1 / sample_rate

        fftMatrix = np.zeros((number_of_blocks, int(blockSize/2)))

        freq = np.fft.fftfreq(blockSize, d = time_steps)

        freq = freq[range(int(blockSize/2))]/1000



        for i in range(0, 4):
                Y = np.fft.fft(block_4096_matrix[i, :]).real / blockSize
                fftMatrix[i, :] = Y[range(int(blockSize / 2))]



        fftMatrix = 20 * np.log10(np.abs(fftMatrix))

        return fftMatrix, freq




