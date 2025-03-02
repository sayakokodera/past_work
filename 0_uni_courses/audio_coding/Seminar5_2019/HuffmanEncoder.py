import collections
from HeapNodes import HeapNodes
import heapq

class HuffmanCoding:

    def __init__(self):
        self.heap = []
        self.codes = {} # key = quant_index, value = binary code
        self.reverse_mapping = {} # key = binary code, value = quant_index


    def findFrequency(self, data):

        frequency = collections.Counter(data)
        return frequency

    def coding(self, quant_index):
        # Count the frequency of each element
        frequency = self.findFrequency(quant_index)
        # Put each element into a heap (= priority queue)
        self.makeNodes(frequency)
        # Construct a Huffman tree using the heap
        self.formTree()
        # Top down assignment of bits
        self.makeCodes()
        # Convert the quantized data into bit stream
        encoded_data = self.encodeData(quant_index)
        # Zero padding to make the bit stream as a set of bytes
        encoded_data_padded = self.addPadding(encoded_data)
        bytes = self.convertDataToByteArray(encoded_data_padded)

        return self.reverse_mapping , bytes

    def formTree(self):

        while (len(self.heap) > 1):
            # heappop : extract the smallest -> delete it from the queue
            # => the first element is always the smallest in the queue
            node1 = heapq.heappop(self.heap) 
            node2 = heapq.heappop(self.heap)
            # Create a HeapNodes object for the parent node
            merged = HeapNodes(None, node1.freq + node2.freq)
            # Register the children nodes in the object
            merged.left = node1
            merged.right = node2
            # Add the parernt node to the heap
            heapq.heappush(self.heap, merged)


    def makeNodes(self, frequency):

        for key in frequency:
            node = HeapNodes(key, frequency[key])
            heapq.heappush(self.heap, node) # min heap
            # the first element = smallest
            # else = random

    def makeCodes(self):

        root = heapq.heappop(self.heap) # Only the sum(frequency) is left in heap
        current_code = ""
        self.helpAssignLeftRightCodes(root, current_code)

    def helpAssignLeftRightCodes(self, root, current_code):

        if root is not None:

            if root.char is not None: # = end of the branch
                self.codes[root.char] = current_code
                self.reverse_mapping[current_code] = root.char
            
            else: # = merged point
                self.helpAssignLeftRightCodes(root.left, current_code + "0")
                self.helpAssignLeftRightCodes(root.right, current_code + "1")

    def encodeData(self, data):
        encoded_data = ""
        for character in data:
            encoded_data += self.codes[character]
        return encoded_data

    def addPadding(self, encoded_data):

        padding = 8 - len(encoded_data) % 8

        for i in range(padding):
            encoded_data += "0"

        # Let the decoder know how many bits to remove at the end of bit streams
        padded_info = "{0:08b}".format(padding) 
        # Insert the padding info as the first byte
        encoded_data = padded_info + encoded_data 

        return encoded_data

    def convertDataToByteArray(self, data):

        bytes = bytearray()

        for i in range(0, len(data), 8):
            byte = data[i:i+8]
            bytes.append(int(byte, 2))
        return bytes

