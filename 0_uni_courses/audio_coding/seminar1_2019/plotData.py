import matplotlib.pyplot as plt
import numpy as np
from wav_file_read_play import wavOperations

class plotData:

    def __init__(self):
        self.wave = wavOperations()
        self.waveArray = None
        self.sample_rate = None
        self.counter = 1

    def plotSingleChannel(self, singleChannelData, sample_rate, title, xlabel, ylabel):

        time = np.linspace(0, len(singleChannelData)/sample_rate, len(singleChannelData))
        plt.figure(self.counter)
        plt.clf()
        plt.plot(time, singleChannelData)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.show()
        self.counter +=1

    def plotLeftRightChannel(self, leftChannelData, rightChannelData, sample_rate, title, xlabel, ylabel):

        time = np.linspace(0, len(leftChannelData)/sample_rate, len(leftChannelData))
        plt.figure(self.counter)
        plt.clf()
        plt.plot(time, leftChannelData, label='left channel')
        plt.plot(time, rightChannelData, label='right channel')
        plt.legend(loc = 'upper right')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.counter += 1
        #plt.show()

    def plotOriginalReconstructedData(self, original_data, reconstructed_data, sample_rate, title, xlabel, ylabel, style = None,
                                      title1 = None, title2 = None):

        time = np.linspace(0, len(original_data) / sample_rate, len(original_data))

        plt.figure(self.counter)
        plt.clf()
        if style == 'singlePlot':
            plt.plot(time, original_data[:,1], label='Original data')
            plt.plot(time, reconstructed_data[:,1], label='Reconstructed data')
            plt.legend(loc='upper right')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        elif style == 'seperatePlot':

            plt.subplot(121)
            plt.plot(time, original_data[:,0], label='Original data')
            plt.title(title1)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.subplot(122)
            plt.plot(time, original_data[:,0], label='Reconstructed data')
            plt.title(title2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        self.counter += 1

    def plotFFTMatrix(self,freq, fftMatrix,xlabel,ylabel,title=None):

        plt.figure(self.counter)
        plt.clf()
        plt.plot(freq, fftMatrix[0, :], 'r', freq, fftMatrix[1, :], 'b', freq, fftMatrix[2, :], 'g', freq,
                 fftMatrix[3, :], 'darkviolet')
        plt.legend(('First 1024 sample', 'Second 1024 sample', 'Third 1024 sample', 'Fourth 1024 sample'))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        self.counter += 1

    def show(self):

        plt.show()



