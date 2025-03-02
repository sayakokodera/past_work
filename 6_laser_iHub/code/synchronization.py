import numpy as np
import pandas as pd
import polars as pl
import os
import h5py  # for loading the IDMT data
from sklearn.decomposition import FastICA
import scipy
import scipy.signal as scsig
from laser_dataset import FileReader # for loading the QASS data

r"""
### Steps for synchronization:
- (1) Load the synchro signals from Dewetron
- (2) Load the IDMT data, ch3 <br>
- (3) Interpolate the Dewetron signal (to compensate the different sampling rate)
    - (3-0) ICA to increase the SNR
    - (3-1) Linear interpolation
- (4) Cross-correlate the interpolated Dewetron signal and the IDMT data

### Note:
For the trials 27-29, ICA cannot be performed, as the dewetron signal is too short (2s).
Use the raw signal for interpolation instead. 
"""

class Synchronizer():
    
    def __init__(self):
        # Sampling frequencies
        self.fs_dew = 10.0*10**3 #[Hz] for Dewetron
        self.fs_idmt = 204800.0 # [Hz] for IDMT
        self.fs_izfp = 97656.0 # [Hz] for IZFP
        # Synchronization signals
        self.T_synch = 1 # Cycle duration of the synch signal
        self.N = int(self.T_synch*self.fs_dew) # number of samples in T_synch for the original Dewetron signal
    
    # ----------------------------
    # IDMT data
    # ----------------------------
    def load_hd5(self, _path, _fname):
        """
        The structure of the IDMT data
            * each file is a nested dictionary
                * key 1 = samurai_export
                * key 2 = Signal
                * key 3 = channel_group_1
                * key 4 = [channel_1, channel_2, channel_3]
                * key 5 = block_1
            * synchronization signal = Ch.3
        """
        f = h5py.File(f'{_path}/{_fname}__iHub2023_.h5', 'r+') 
        dset = f['samurai_export']['Signal']['channel_group_1']
        # Load the synchronization signal
        s = dset['channel_3']['block_1'][:].squeeze() # adding [:] returns a numpy array
        # Zero-mean and normalize the signal
        s = s - np.mean(s)
        s = s/np.abs(s).max()
        # Time
        T = 1/self.fs_idmt* len(s)
        t = np.arange(-1.0, T - 1.0, 1/self.fs_idmt)
        return (t, s)
    
    # ----------------------------
    # IZFP data
    # ----------------------------
    def load_qass(self, _path, _qassNo, standardize=True):
        # Setting up the file name: Ch.4 is the synch signal
        fname_prefix = '20231016_TU_Ilmenau_Process'
        fname_suffix = 'SIG_Raw_compress_1'
        fname = f'{_path}/{fname_prefix}_{_qassNo}_Ch4_{fname_suffix}'
        # Instantiate the reader
        reader = FileReader()
        # Load the data 
        s = reader.load_data(f'{fname}.bin')
        # Zero-mean and normalize the signal
        if standardize == True:
            s = s - np.mean(s)
            s = s/np.abs(s).max()
        # Time
        t = 1/self.fs_izfp* np.arange(0, len(s)) # No pre-trigger included
        return (t, s)
    
    
    # ----------------------------
    # Dewetron data
    # ----------------------------  
    def load_dew(self, _path, _fname, sigtypes=['Sync_V']):
        # Lazy loading
        q = (pl.scan_csv(f'{_path}/{_fname}.csv', has_header=True))
        # Time 
        t = q.select(['Time_s']).collect().to_numpy().squeeze() # time in [s]
        t = np.around(t, 6) # remove the rounding error
        # Signal
        S = np.zeros((len(t), len(sigtypes)))
        for col, key in enumerate(sigtypes):
            S[:, col] = q.select([key]).collect().to_numpy().squeeze() # synchronization signal [V]
        if len(sigtypes) == 1:
            S = S.squeeze()
        return (t, S)
    
    # ---------------------------------------
    # Option 1: interpolate the Dewetron data
    # ---------------------------------------
    def denoise_ica(self, s):
        """
        Parameters
        ----------
            s: np.array
                Synch signal from Dewetron which needs to be denoised
                The original signal should be loonger than 3s (i.e. 3* T_synch)
                
        Returns
        -------
            s_denoised: np.array
                Denoised signal with the duration of 1s (= T_synch)
        """
        # Zero-mean and normalize the signal
        s = s - np.mean(s)
        s = s/np.abs(s).max()
        # Reformulate into an array: 3 collumns of 1s (=T_synch) vectors
        S = np.array([s[:self.N], s[self.N:2*self.N], s[2*self.N:3*self.N]]).T
        # ICA
        transformer = FastICA(n_components=2, random_state=0, whiten='unit-variance')
        S_ica = transformer.fit_transform(S)
        # Denoised signal
        s_denoised = S_ica[:, 0] # -> duration = 1s (+ T_synch)
        # Normalize
        s_denoised = s_denoised/np.abs(s_denoised).max()
        return s_denoised
    
    def interpolate(self, t, s, t_new_long, fs_new):
        """
        Parameters
        ----------
            t: 1D array [s]
                Time points of the original synchronization (Dewetron) data 
            s: 1D array 
                Dewetron signal to be interpolated
            t_new_long: 1D array (longer than t_old)
                Time points taken with a higher sampling rate than t
            fs_new: float [Hz]
                New sampling rate 
        """
        # Check if the provided signal s is longer than 1s (= T_synch)
        # -> if not, the interpolator cannot compute the value for t=T_synch
        # -> solution: repeat the vector s 
        if len(s) < len(t):
            reps = int(len(t)/self.N) # = numer of repetitions
            s = np.tile(s, reps) 
        # Instantiate the interpolator: provide the original points 
        f = scipy.interpolate.interp1d(t[:len(s)], s, kind='linear')
        # Time points to be interpolated 
        t_interp = t_new_long[:int(self.T_synch*fs_new)] # -> duration = T_synch
        # Intepolation
        s_interp = f(t_interp)
        return (t_interp, s_interp)
    
    
    def load_and_process_dew(self, _path, _fname, t_new, fs_new, sigtypes=['Sync_V']):
        # Load
        t_dew, s_dew = self.load_dew(_path, _fname, sigtypes)
        # Denoise
        s_ica = self.denoise_ica(s_dew)
        # Interpolate
        t_interp, s_interp = self.interpolate(t_dew, s_ica, t_new, fs_new)
        return (t_interp, s_interp)
        
    # ----------------------------------------------
    # Option 2: downsample the SoundBook / QASS data
    # ----------------------------------------------
    def lowpass(self, s, fs):
        """
        Lowpassing the synchsignal prior to downsampling to avoid aliasing. 
        
        Parameters
        ----------
            s: array (time domain)
                Synchsignal of either SoundBook or QASS, which needs to be downsampled
            fs: float [Hz]
                Sampling frequency of s
        """
        #----- (1) Filter design
        # (1.1) Construct an analog filter (Butterworth) 
        filts_lp = scsig.lti(*scsig.butter(N=5, Wn=2*np.pi* 0.5* self.fs_dew, btype='lowpass', analog=True))
        # (1.2) Convert (1) into a digital filter using bilinear transform
        filtz_lp = scsig.lti(*scsig.bilinear(b=filts_lp.num, a=filts_lp.den, fs=fs))
        # (1.3) Compute the freq. responses: w = angular freq., h = freq response
        wz_lp, hz_lp = scsig.freqz(filtz_lp.num, filtz_lp.den, worN=int(len(s)/2)) # = freq. response of (2), i.e. digital
        # (1.4) Extend the filter to both frequency sides
        hz_both = np.concatenate((hz_lp, np.flip(hz_lp)))
        wz_both = np.concatenate((0.5*wz_lp, 0.5*np.pi+0.5*wz_lp))
        #----- (2) Apply the filter
        # (2.1) Frequency response
        sp = np.fft.fft(s)
        freq = fs*np.fft.fftfreq(len(sp))
        sp_lp = hz_both* sp
        # (2.2) Time domain signal
        s_lp = np.fft.ifft(sp_lp).real
        return s_lp
    
    def downsample(self, s, t, fs, ret_signal_only=True):
        """
        Downsampling the synchsignals of either SoundBook or QASS.
        
        Parameters
        ----------
            s: array (time domain)
                Synchsignal of either SoundBook or QASS, which needs to be downsampled
            t: array in [s]
                Measurement time points associated to s (required for resampling)
            fs: float [Hz]
                Sampling frequency of s (required for lowpass)
        """
        #----- (1) Lowpassing to avlid aliasing
        s_lp = self.lowpass(s, fs)
        
        #----- (2) Down sampling
        # Number of samples in the resampled signal needs to be specified
        # Because the durations of the Dewetron and another system are not the same
        num_resamp = int((t[-1] - t[0])* self.fs_dew)
        # Dwonsample
        s_ds, t_ds = scsig.resample(s_lp, num=num_resamp, t=t)
        fs_ds = len(s_ds)/(t_ds[-1]-t_ds[0])
        
        if ret_signal_only == True:
            return s_ds
        else:
            return s_ds, t_ds, fs_ds
    
    # ----------------------------
    # Identify the delay:
    # -> cross-correlate the signals -> peak finding
    # ---------------------------- 
    def identify_delay(self, s_dew, s, fs, ret_corr=False):
        """
        Parameters
        ----------
            s_dew: 1d array (!!!! duration = T_synch !!!!)
                Interpolated Dewetron signal
                !!! Very important here is that duration of this signal needs to be T_synch !!!
            s: 1d array
                Signal whose delay needs to be corrected (from SoundBook or QASS)
            fs: float [Hz]
                Sampling frequency of s
        """
        # Number of samples within T_synch (based on the provided sampling frequency fs)
        N_synch = int(self.T_synch*fs)
        # Cross-correlate: len(corr) = len(s_dew) + len(s) - 1
        corr = scsig.correlate(s, s_dew, mode='full', method='direct')
        # Adjust the x-axis (i.e. time shift) of the correlation
        tau = np.around(1/fs* np.arange(0, len(corr)) - self.T_synch + 1/fs, 9)
        
        # First peak of the cross correlation in -0.5s < tau < 0.5s
        idx_peak = np.argmax(corr[int(0.5*N_synch):int(1.5*N_synch)]) + int(0.5*N_synch)
        # Delay = peak point
        delay = tau[idx_peak]
        N_offset = int(delay*fs) # Samples to skip 
        
        if ret_corr == False:
            return (delay, N_offset)
        else:
            print('Synchronizer: returning the correlation results')
            return (delay, N_offset, corr, tau)
        
    