import numpy as np


class HuffmanDecoding:

    def getPaddedTextFromByteArray(self, bytearray):
        bit_string = ""
        for byte in bytearray:
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string += bits
        return bit_string

    def removePadding(self, paddedEncodedText):
        paddedInfo = paddedEncodedText[:8]
        # bits -> decimal
        extraPadding = int(paddedInfo, 2)
        # Remove the padding info at the beginning
        paddedEncodedText = paddedEncodedText[8:]
        # Remove the padded zeros at the end
        encodedText = paddedEncodedText[: -1 * extraPadding]
        return encodedText

    def reverseMapping(self, encodedText, reverse_mapping):
        currentText = ""
        decodedUintArray = np.array([], dtype=np.uint8)

        for bit in encodedText:
            currentText += bit
            if(currentText in reverse_mapping):
                character = reverse_mapping[currentText]
                decodedUintArray = np.append(decodedUintArray, character)
                currentText = ""
        return decodedUintArray

    def decompress(self, bytearray, reverse_mapping):
        
        # Convert bytes to bits
        paddedString = self.getPaddedTextFromByteArray(bytearray)
        # Remove the padding info and the padded zeros
        encodedString = self.removePadding(paddedString)
        # Convert the bit stream to the quantization indexes
        indexes = self.reverseMapping(encodedString, reverse_mapping)

        return indexes
