# -*- coding: utf-8 -*-
"""
Audio Coding: Seminar 1, task 3 & 4 (Quantization)

@author: Sayako Kodera
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from encframewk import Encoder
from decframewk import decoder


### Encoding ###
fnameinput = 'Track16.wav'
enc = Encoder()
enc.read_file(fnameinput)
enc.normalize_wavdata()
# 16 bits
fnamebin_16 = 'encoded.bin'
enc.encode(bits = 16, fnameoutput = fnamebin_16)
wavdata_org = enc.get_wavdata(normalize = True)
# 8 bits
fnamebin_8 = 'encoded8bit.bin'
enc.encode(bits = 8, fnameoutput = fnamebin_8)

# Compare the data size
print('***Size of each data***')
print('Original wav data: {}MB'.format(os.stat(fnameinput).st_size/10**6))
print('16bits quantized bin data: {}MB'.format(os.stat(fnamebin_16).st_size/10**6))
print('8bits quantized bin data: {}MB'.format(os.stat(fnamebin_8).st_size/10**6))


### Decoding ### -> @Atul: feel free to change here
# 16bits
stepsize16 = 2 / (2**16)
dataqraw_16 = pickle.load(open(fnamebin_16, 'rb'))
dataq_16 = dataqraw_16* stepsize16

# 8 bits
stepsize8 = 2 / (2**8)
dataqraw_8 = pickle.load(open(fnamebin_8, 'rb'))
dataq_8 = dataqraw_8* stepsize8


### Plotting ###
samplingrate = enc.samplingrate
timescale = np.linspace(0, len(wavdata_org) / samplingrate, len(wavdata_org))

plt.figure(1)
plt.plot(timescale, wavdata_org[:, 1], label = 'Original data')
plt.plot(timescale, dataq_16[:, 1], '--', label = 'Quantized data (16bits)')
plt.plot(timescale, dataq_8[:, 1], '--', label = 'Quantized data (8bits)')
plt.legend()
plt.title('Original vs quantized signal')
plt.xlabel('Time')
plt.ylabel('Normalised Amplitude')

plt.figure(2)
plt.plot(timescale, wavdata_org[:, 1], label = 'Original data')
plt.plot(timescale, dataq_16[:, 1], '--', label = 'Quantized data (16bits)')
plt.legend()
plt.title('Original vs 16bits quantization')
plt.xlabel('Time')
plt.ylabel('Normalised Amplitude')

plt.figure(3)
plt.plot(timescale, wavdata_org[:, 1], label = 'Original data')
plt.plot(timescale, dataq_8[:, 1], '--', label = 'Quantized data (8bits)')
plt.legend()
plt.title('Original vs 8bits quantization')
plt.xlabel('Time')
plt.ylabel('Normalised Amplitude')

plt.show()
