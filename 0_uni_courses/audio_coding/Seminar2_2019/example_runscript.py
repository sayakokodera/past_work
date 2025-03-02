# -*- coding: utf-8 -*-
"""
AC Sminar2: example runscript
"""
import numpy as np
import matplotlib.pyplot as plt
import demo_moodle.psyacmodel as ac
import scipy.signal as sc

plt.close('all')
#================================================================================================ Parameter Setting ===#
# Freq. domain
fS = 44100 # sampling freq. [Hz]
fmax = fS/2 # Nyquist frequency
Nfft = 1024 # Number of subbands in STFT domain
f0 = fmax / Nfft # width of the DFT subband
# Bark domain
Nbark = 2*24 # Number of subbands in Bark domain
barkmax = ac.hz2bark(fmax) # max in Bark domain, corresponding to fmax
bark = np.linspace(0, barkmax, Nbark)
W = ac.mapping2barkmat(fS, Nbark, 2*Nfft) # matrix for converting Hz into Bark 
W_inv = ac.mappingfrombarkmat(W, 2*Nfft) # inverse: Bark -> Hz

# Spreading fct relevant
alpha = 0.8

#========================================================================================================== Signal ===#
# Signal
t = np.arange(3*60*fS)/fS
tone = np.sin(2*np.pi*200*f0*t) + np.sin(2*np.pi*600*f0*t)

# STFT
f, _, toneSTFT = sc.stft(tone, fS, nperseg = 2*Nfft)
toneSTFT = np.abs(toneSTFT[:Nfft, :]) #????? -> becomes the input for the masking threshold
spectrum_FT = np.sum(toneSTFT, axis = 1)/toneSTFT.shape[1]
plt.figure(1)
plt.plot(spectrum_FT)

# Bark domain
spectrum_Bark = ac.mapping2bark(spectrum_FT, W, 2*Nfft)
plt.figure(2)
plt.plot(spectrum_Bark)


#==================================================================================================== Spreading fct ===#
spfctBarkdB = ac.f_SP_dB(fmax, Nbark)
plt.figure(3)
plt.plot(bark, spfctBarkdB[26:(26+Nbark)])
plt.axis([6, 23, -100, 0])
plt.title('Spreading Function')

spfctmat = ac.spreadingfunctionmat(spfctBarkdB, alpha, Nbark)
plt.figure(4)
plt.imshow(spfctmat)

#=============================================================================================== Masking threshold ===#
mthToneBark = ac.maskingThresholdBark(spectrum_Bark, spfctmat, alpha, fS, Nbark) #[0, :] = mth for tone, [1, :] = LTQ
mthToneFreq = ac.mappingfrombark(mthToneBark, W_inv, 2*Nfft) #[0, :] = mth for tone, [1, :] = LTQ
delta = mthToneFreq[0, :]*2


plt.figure(5)
plt.plot(20*np.log10(mthToneBark[0, :] + 10**(-6)), label = 'excl. LTQ')
plt.plot(20*np.log10(np.maximum(mthToneBark[0, :], mthToneBark[1, :]) + 10**(-6)), label = 'incl. LTQ')
plt.title('Masking Threshold in Bark')
plt.legend()
plt.xlabel('1/2 Bark subband')
plt.ylabel('[dB]')

plt.figure(6)
plt.plot(20*np.log10(mthToneFreq[0, :] + 10**(-6)), label = 'excl. LTQ')
plt.plot(20*np.log10(np.maximum(mthToneFreq[0, :], mthToneFreq[1, :]) + 10**(-6)), label = 'incl. LTQ')
plt.title('Masking Threshold in Linear Domain')
plt.legend()
plt.xlabel('Freq. subband')
plt.ylabel('[dB]')

