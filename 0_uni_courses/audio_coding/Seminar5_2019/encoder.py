from HuffmanEncoder import HuffmanCoding
import pickle
from wav_file_read_play import wavOperations
from quantization import Quantize


class Encoder:

    def __init__(self):

        self.wav = wavOperations()
        self.number_of_bits = 8

    def encoder(self):

        encoded_data_dict = {}
        sample_rate, wavArrayData = self.wav.getWavParameters("Track16.wav")
        number_of_channels = wavArrayData.shape[1]

        for i in range(number_of_channels):

            huff = HuffmanCoding()
            
            # Uniform quantization
            encoder = Quantize(wavArrayData[:120000, i], sample_rate)
            quant_index = encoder.encode(self.number_of_bits)
            step_size = encoder.get_step_size()

            # Huffman Encoding
            reverse_mapping , bytes = huff.coding(quant_index)
            #above line can be modified with simple data by keeping e.g. quant_index = [1,2,4,5,2,3,4] to understand it.

            # Form a dictionary to dump 
            encoded_data_dict['channel' + str(i)] = {'bytes': bytes, 'step_size': step_size, 'reverse_mapping': reverse_mapping}

        pickle.dump(encoded_data_dict, open("encoded.bin", "wb"), pickle.HIGHEST_PROTOCOL)
