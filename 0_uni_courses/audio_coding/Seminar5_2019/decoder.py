from HuffmanDecoder import HuffmanDecoding
import pickle
import numpy as np
from dequantization import Dequantize

class Decoder:

    def decoder(self, size):

        encoded_data_dict = pickle.load(open("encoded.bin", "rb"))

        number_of_channels = len(encoded_data_dict.keys())

        huffDec = HuffmanDecoding()

        reconstructed_data_merged = np.zeros(size)

        for i in range(number_of_channels):

            bytes = encoded_data_dict['channel'+str(i)]['bytes']
            step_size = encoded_data_dict['channel'+str(i)]['step_size']
            reverse_mapping = encoded_data_dict['channel'+str(i)]['reverse_mapping']

            indexes = huffDec.decompress(bytes, reverse_mapping)

            deq = Dequantize()
            deq.decode(indexes, step_size)
            reconstructed_data = deq.getReconstructedData()

            reconstructed_data_merged[: ,i] = reconstructed_data

        return reconstructed_data_merged
