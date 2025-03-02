import matplotlib.pyplot as plt



class PlotResults:


    def __init__(self):

        self.counter=1


    def plotRamp(self,ramp ):


        plt.figure(self.counter)
        plt.plot(ramp)
        plt.title('Input Signal - Ramp')
        plt.xlabel('Sample')
        plt.ylabel('Output')
        self.counter = self.counter +1


    def plotReconstrRamp(self, original_ramp, reconstructed_ramp):

        plt.figure(self.counter)
        plt.plot(original_ramp, 'b', label = 'Original')
        plt.plot(reconstructed_ramp, 'r', label ='Reconstructed')
        plt.xlabel('samples')
        plt.ylabel('amplitude')
        plt.title('Original and Reconstructed Ramp signal')
        plt.legend()
        self.counter = self.counter +1

    def plotReconstrMono(self, original_mono, reconstructed_mono):

        plt.figure(self.counter)
        plt.plot(original_mono, 'b', label='Original signal')
        plt.plot(reconstructed_mono, 'r', label='Reconstructed signal')
        plt.xlabel('samples')
        plt.ylabel('amplitude')
        plt.title('Original and Reconstructed audio signal')
        plt.legend()
        self.counter = self.counter + 1


    def plotFirstandLastSubband(self, subband1, subband2):

        plt.figure(self.counter)
        plt.plot(subband1, 'b', label='First Subband ')
        plt.plot(subband2, 'r', label='Last Subband ')
        plt.xlabel('samples')
        plt.ylabel('amplitude')
        plt.title('First Subband and Last Subband')
        plt.legend()
        self.counter = self.counter + 1

    def show(self):
        plt.show()

