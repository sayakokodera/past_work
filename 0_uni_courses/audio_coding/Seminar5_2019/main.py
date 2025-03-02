from wav_file_read_play import wavOperations
import matplotlib.pyplot as plt
from encoder import Encoder
from decoder import Decoder

if __name__=='__main__':

    wav = wavOperations()
    sample_rate, wavArrayData = wav.getWavParameters("Track16.wav")
    size = wavArrayData[:120000, ].shape
    number_of_channels = wavArrayData.shape[1]

    encoder = Encoder()
    decoder = Decoder()
    encoder.encoder()

    reconstructed_data_merged = decoder.decoder(size)

    for i in range(reconstructed_data_merged.shape[1]):

        plt.figure(i+1)
        plt.title('Reconstruction with 8 bits for Channel ' + str(i+1))
        plt.plot(wavArrayData[:120000, i], label = 'Original signal')
        plt.plot(reconstructed_data_merged[:, i], label = 'Reconstructed signal')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.show()
    #wav.playExtractedData(reconstructed_data_merged, sample_rate, number_of_channels)




