import numpy as np
import pandas as pd
import scipy

from laser_dataset import LaserDataInteractor
from tools.file_writer import save_data


class Segmenter():
    """
    Example usage
    -------------
        segmentor = Segmenter(meta_df)
        segmentor.set_path(path2load='XXX/YYY/ZZZ', path2save='AAA/BBB/CCC')
        segmentor.set_segmentation_params(fs=6.25*10**6, past=0.0 future=0.0)
        
        # 1st trial
        segmenter.trialNo = '1_2'
        ch1, dt1 = segmenter.load(chNo=1, clip=True)
        segmenter.segment_and_save(ch1)
        
        # 2nd trial
        segmenter.trialNo = '1_8'
        ch1, dt1 = segmenter.load(chNo=1, clip=True)
        segmenter.segment_and_save(ch1)
        
        ...
        
    """
    
    def __init__(self, _meta_df):
        # Copy
        self._meta_df = _meta_df
        
        # Initialize
        self.fs = None
        self.past = None
        self.future = None
        self.path2load = None
        self.path2save = None
        
        
    
    
    # ----------------------------
    # Setter & meta data handling
    # ----------------------------
    def set_path(self, path2load, path2save):
        self.path2load = path2load
        self.path2save = path2save
        
    def set_segmentation_params(self, fs, past=0.0, future=0.0):
        """
        Parameters
        ----------
            fs: float in [Hz]
                Sampling frequency 
        """
        self.fs = fs
        self.past = past
        self.future = future
        
        
    @property
    def trialNo(self):
        return self._trialNo
    
    @trialNo.setter
    def trialNo(self, trialNo, verbose=False):
        if trialNo in self._meta_df['trial'].unique():
            self._trialNo = trialNo
            if verbose == True:
                print(f'Segmenter: trialNo is set to {trialNo}')
        else:
            raise ValueError(f'trialNo {trialNo} is not in the meta data')
        
    
    def meta_data_handling(self):
        # Dataframe row of the desired trial
        _df_row = self._meta_df[self._meta_df['trial'] == self.trialNo].reset_index(drop=True)
        
        # Parameters for data loading
        #self.qassNo = str(int(_df_row['QASS'].values[0])).zfill(3) # e.g. 40 -> 040
        self.qassNo = str(_df_row['QASS'][0]).zfill(3) # e.g. 40 -> 040
        
        # Parameters for segmenting
        self.dataID_arr = np.array(_df_row['dataID'])
        self.t_arr = np.array(_df_row['t[s]']) #[s]
        self.T_seg = 10**-3* (_df_row['T_seg[ms]'][0]) #[s]
        
    
    
    # ----------------------------
    # Load the entire signal
    # ----------------------------
    def load(self, chNo, clip=True, verbose=False):         
        # Setup: chNo, abslimit, path2load
        self.chNo = chNo
        self.abslimit = clip
        
        self.comments(
            text=f'Segmentor: load {self.trialNo}, ch{self.chNo}!', 
            verbose=verbose
        )
        
        # Load using the interactor
        interactor = LaserDataInteractor()
        sig = interactor.load(chNo=self.chNo, path=self.path2load, qassNo=self.qassNo)

        # Clip & normalize the signal -> to mitigate the effect of outliers
        # if self.abslimit != None:
        #     sig = np.clip(sig, a_min = -self.abslimit, a_max = self.abslimit)/self.abslimit
            
        return sig
            
        
    @property
    def abslimit(self):
        return self._abslimit
    
    @abslimit.setter
    def abslimit(self, clip):
        if clip == False:
            self._abslimit = None
        else:
            # Ver. 23.01.06
            # abslimit_ch1 = 2094.5, abslimit_ch2 = 1762.5, abslimit_ch3 = 1272.0
            if self.chNo == 1:
                self._abslimit = 2094.5
            elif self.chNo == 2:
                self._abslimit = 1762.5
            elif self.chNo == 3:
                self._abslimit = 1272.0
            else:
                raise ValueError(f'chNo = {chNo} is invalid!')
                
    
    # ----------------------------
    # Segment the signal 
    # ----------------------------
    def segment_and_save(self, sig, verbose=False):
        """
        This function does two things:
            (1) Pick the desired segment from the full length data
            (2) Resample the segment with the desired sampling freq.
                => s.t. all segments have the same length! 

        Parameters
        ----------
            sig: array, 1D
                Full-length data (single channel)
            verbose: boolean

        """
        # Params
        # Actual duration of each segment including with the past and the future
        T = self.T_seg + self.future + self.past # [s]
        # Segment length as a number of samples
        length = int(T* self.fs) 
        
        # All start & end points for segmentation (index!)
        starts = ((self.t_arr - self.past)* self.fs).astype(int) # indices!
        ends = starts + length # indices!
        
        # Array containing info about start, end and dataID
        info_arr = np.concatenate((starts[:, np.newaxis], ends[:, np.newaxis], self.dataID_arr[:, np.newaxis]), axis = 1) 
        
        # Save
        np.apply_along_axis(
            func1d=self._save, axis = 1, arr=info_arr, sig=sig, verbose=verbose
        )
    
    # ----------------------------
    # Save each segment 
    # ----------------------------
    def _save(self, info, sig, verbose):
        """
        Parameters
        ----------
            info: tuple of float, len = 3
                Containing the info regarding the current segment
                * start: starting point of the segment, float in [s]
                * end: starting point of the segment, float in [s]
                * dataID: to save the segment, int
            sig: array
                Entire signal 
            verbose: boolean
        """
        # Unzip
        (start, end, dataID) = info
        
        # Segment & save
        save_data(
            data = sig[start:end], 
            path = f'{self.path2save}/ch{self.chNo}', 
            fname = f'data_{dataID}.npy', 
            notify = False
        )
        self.comments(
            text=f'Sagment saved for trialNo = {self.trialNo}, ch{self.chNo}, dataID = {dataID}!', 
            verbose=verbose
        )
        
        
    # ----------------------------
    # Miscellaneous
    # ----------------------------
    def comments(self, text, verbose):
        if verbose == True:
            print(text)