#==============================================================
# Data annotator class for the measurement campaign iHub 2023
#==============================================================

import numpy as np
import pandas as pd
import warnings


class MetadataGenerator():
    """
    Example usage
    -------------
        metagen = MetadataGenerator()
        metagen.set_constants(
            versuche_df = versuche, 
            delay_df = delay_df,
            T_seg=120*10**-3, 
            t_buffer=20*10**-3, 
            incl_transition=False,
            binary=True,  
            steps=None, 
        )
        
        # Base of the metadata
        metadata = {}
        
        # For the first trial: 1_5
        metagen.trialNo = '1_5'
        metadata.update(metagen.generate_trial_metadata(buffer=0))
        
        # For the second trial: '2_10'
        metagen.trialNo = '2_10'
        metadata.update(metagen.generate_trial_metadata(buffer=len(metadata)))
        
        ...
    """
    
    def __init__(self):
        # Instantiate helper classes
        self.annotator = SegmentAnnotator()
        self.segprovider = SegmentPointsProvider()
        
        #*** global attributes***
        # inter-trial constants
        self.T_seg = None
        # parameters unique for each trial
        #self.trialNo = None
        self.df_row = None
        self.t_patches = None
        self.seg_points = None
        
    # ----------------------------
    # Inter-trial constants
    # ----------------------------    
    def set_constants(self, versuche_df, delay_df,
                      T_seg, t_buffer, 
                      incl_transition, binary, 
                      steps=None, patches=[1, 2, 3, 4, 5], 
                      class_bins=None):
        """
        Setting the value for arguments which remain constant for all trials.
        
        Inter-trial constants:
        ----------------------
            versuche_df: pandas df
                Dataframe version of the Versuchstabelle
            delay_df: pandas df
                Dataframe containing the information on the system delays
            T_seg: flot [s]
                Segment duration, !! global attribute !!
            time_buffer: float [s]
                Buffer to start segmenting. 
                If the transitions are excluded, this time_buffer is applied to the beginning of
                all patches. 
            incl_transition: boolean
                True, if the transitions should be included for segmenting
            binary: boolean
                True, if binary classification is desired
            steps: float [s]
                interval between two segments
                None by default (and in this case steps = T_seg)
            patches: list
                List of patches which are segmented
                All segments (= [1, 2, 3, 4, 5]) by default
                (e.g.) 
                    [1, 3, 5] -> segmenting only the on-weld patches
                    'N' = patch including only the ambient noise, taken after P5 
            class_bins: list/array, [mm]
                Bin edges of the classes, requred if the transitions are included
                None by default

        """
        # Global attributes
        self.T_seg = round(T_seg, 9) #[s]
        
        # Dataframes
        self.df = versuche_df.copy()
        self.delay_df = delay_df.copy()
        
        # Constants for segprovider
        self.segprovider.set_constans(T_seg, t_buffer, binary, incl_transition, steps, patches)
        
        # Constants for annotator
        self.binary = binary 
        self.class_bins = class_bins
        

    # ----------------------------
    # Dataframe for the current trial
    # ----------------------------
    @property
    def trialNo(self):
        return self._trialNo
    
    @trialNo.setter
    def trialNo(self, trialNo):
        # Set the 
        self._set_trial_params(trialNo)
    
    # -> v.23/11/22: adjusted to the iHub 2023 campaign
    def _set_trial_params(self, trialNo):
        self._trialNo = trialNo
        # Current data frame row
        self.df_row = self.df.loc[self.df['DEWETRON'] == self.trialNo]

    # ----------------------------
    # Metadata for the selected trial
    # ----------------------------
    def generate_trial_metadata(self, buffer):
        # Setup the segment points
        self.compute_segment_points()
        # Setup the annotator
        self.annotator.extract_gap_info(self.df_row)
        
        # Base of the trial metadata
        trial_meta = {}
    
        # Iterate over each segment
        for seg_idx, t in enumerate(self.seg_points):
            # ID to store the current segment
            dataID = seg_idx + buffer
            # Annotating the current segment
            seg_info = self.annotate_segment(t, dataID, seg_idx)
            # Update the trial metadata
            trial_meta.update({
                dataID: seg_info
            })
            
        return trial_meta
    
    # ----------------------------
    # Segment points
    # -> v.23/11/22: adjusted to the iHub 2023 campaign
    # ----------------------------
    def compute_segment_points(self):
        """
        This function uses SegmentPointsProvider class. 
        For a further detail, see its documentation (example usage) below.
        """
        # (1) Load the necessary info
        t_start_camera = self.df_row['tstart [s]'].item() #[s]
        t_end_camera = self.df_row['tend [s]'].item() #[s]
        delay = self.delay_df['delay[s]'][self.delay_df['DEWETRON'] == self.trialNo].item() #[s]
        
        # (2) Compute t_patches = [t_start, t_p1, t_p2, t_p3, t_p4, t_end]
        self.t_patches = self.segprovider.compute_patch_points(
                        t_start_camera, t_end_camera, delay
                        ) #array, [s]
        # (3) Compute the time points to cut each segment
        self.seg_points = self.segprovider.compute_segment_points(self.t_patches) #array, [s]
        
    # ----------------------------
    # Annotating: varies with each segment
    # -> v.23/11/22: adjusted to the iHub 2023 campaign
    # ----------------------------
    def annotate_segment(self, _t, _dataID, _seg_idx):
        """
        This function uses SegmentAnnotator class. 
        For a further detail, see its documentation (example usage) below.
        """
        # if _t is NOT in the noise range
        if ((self.t_patches[0]<= _t) and (_t <= self.t_patches[-1])):
            ## (1) Determine the patch and the "completeness" ratio
            (patch_idx, ratio) = self.annotator.compute_patch_number(_t, _t+self.T_seg, self.t_patches)
            patch = 'P{}'.format(patch_idx+1)

            ## (2) Calculate the gap (reflecting the ratio as well)
            gap = self.annotator.compute_gap(patch_idx, ratio) #[mm]
            
        # If _t is within the noise range
        else:
            patch = 'N'
            ratio = np.nan
            gap = np.nan
            
        # (3) Identify the appropriate class label 
        classID, class_name = self.annotator.identify_class(
                                gap, 
                                binary=self.binary, 
                                class_bins=self.class_bins
                            ) 
        # Create a dictionary 
        seg_info = {
            'dataID': _dataID, 
            'trial' : self.trialNo,
            'QASS': int(self.df_row['QASS'].values[0]),
            'T_seg[ms]': float(self.T_seg*10**3), # in [ms]
            'segment' : _seg_idx, # int, unitless
            't[s]': _t, # in [s], starting point of the segment
            'patch' : patch,
            'completeness' : ratio, # float in (0, 1], unitless
            'gap[mm]' : gap, # in [mm]
            'classID': classID, # idx
            'class': class_name,
        }
        
        return seg_info
    
    #----------------------------
    # Get the classes
    # ----------------------------
    def get_classes(self):
        # Return the classes in the correct ID order set in the annotator class
        return tuple(self.annotator.classes)
        
    

#========================================================================================
#========================================================================================
class SegmentPointsProvider():
    """
    Comment on 2023/11/21(Tue): 
        This class is already adjusted to the latest measurement campaign iHub 2023. 
        Further modification is not required. 
    
    Example usage
    -------------
        segprovider = SegmentPointsProvider()
        segprovider.set_constans(T_seg, t_buffer, binary, incl_transition, steps, patches)
        
        #------ For the trial 01_0_1_Spalt_pneu
        # (1) Load the necessary info
        df_row = versuche[versuche['DEWETRON'] == '01_0_1_Spalt_pneu']
        t_start_camera1 = df_row['tstart [s]'].item()
        t_end_camera1 = df_row['tend [s]'].item()
        delay1 = delay_df['delay[s]'][delay_df['DEWETRON'] == '01_0_1_Spalt_pneu'].item() #[s]
        # (2) Compute segment points
        t_patches1 = segprovider.compute_patch_points(t_start_camera1, t_end_camera1, delay1) #array, [s]
        seg_points1 = segprovider.compute_segment_points(t_patches1) #array, [s]
        
        #------ For the trial 02_0_1_Spalt_pneu, using the same segprovider
        # (1) Load the necessary info 
        df_row = versuche[versuche['DEWETRON'] == '02_0_1_Spalt_pneu']
        t_start_camera2 = df_row['tstart [s]'].item()
        t_end_camera2 = df_row['tend [s]'].item()
        delay2 = delay_df['delay[s]'][delay_df['DEWETRON'] == '02_0_1_Spalt_pneu'].item() #[s]
        # (2) Compute segment points
        t_patches2 = segprovider.compute_patch_points(t_start_camera2, t_end_camera2, delay2) #array, [s]
        seg_points2 = segprovider.compute_segment_points(t_patches2) #array, [s]
        ...
    """
    
    def __init__(self):
        pass
    
    def set_constans(self, T_seg, t_buffer, 
                     binary, incl_transition, steps, patches,
                     N_patch_noise= 3, start=None, end=None):
        # Setting the inter-trial constants
        self.T_seg = T_seg # [s]
        self.t_buffer = t_buffer # [s]
        self.incl_transition = incl_transition
        if steps is None:
            self.steps = self.T_seg  
        elif steps > self.T_seg:
            warnings.warn(
                'SegmentPointsProvider: the provided step size is too large. ' +
                f'self.steps is thus set to self.T_seg = {round(self.T_seg*10**3, 1)}ms.'#
            )
            self.steps = self.T_seg
        else:
            self.steps = steps
        self.patches = patches
        self.N_patch_noise = N_patch_noise # for deciding the number of segments used for noise
        self.start = start # glpobal starting point of the signal, applied to all trials
        self.end = end # global end point of the signal, applied to all trials
        # Compute the duration of a single patch
        self.T = 160*10**-3 - 2* self.t_buffer # correspond to a whole 32mm width patch
        self.N_seg = round((self.T - self.T_seg)/self.steps) + 1 # number of segments / patch
        
    
    def compute_patch_points(self, t_start_camera, t_end_camera, delay):
        """
        Parameters
        ----------
            t_start_camera: float [s]
                Time point when the weld started in the HD camera.
                This is read from the metadata (Versuchstabelle). 
            t_end_camera: float [s]
                Time point when the weld ended in the HD camera.
                This is read from the metadata (Versuchstabelle).
            delay: float [s]
                System delay between HD camera and QASS system. Read from the 
                delay csv file.
        """
        #---- Synchronization 
        t_start = t_start_camera - delay # weld start in the QASS recording
        t_end = t_end_camera - delay # weld end in the QASS recording 
        #---- Compute the time points of each patch 
        # t_patches = [t_start, t_p1, t_p2, t_p3, t_p4, t_end]
        t_patches = np.linspace(start = t_start, stop = t_end, num = 6) # in [s]
        # Eliminate rounding inaccuracy of python
        return np.around(t_patches, 9) # [s]
    
    
    def compute_segment_points(self, t_patches):
        # Start and end point of the signals
        if self.start is None:
            self.start = t_patches[0]
        # Global end point: should include the noise patches!
        if self.end is None:
            # In case noise patch should be included -> add enough buffer!
            if 'N' in self.patches:
                self.end = t_patches[-1] + 5*self.t_buffer + 3* self.T
            # Otherwise, use the end point of the last patch
            else:
                self.end = t_patches[-1]
        
        if self.incl_transition == True:
            seg_points = self._segment_include_transition(t_patches) # [s]
        else:
            seg_points = self._segment_exclude_transition(t_patches, self.patches) # [s]
        return seg_points
    
    
    def _segment_include_transition(self, t_patches):
        #print(f'Using the entire signal from start={round(start, 6)}s till end={round(end, 6)}s!') 
        return np.around(np.arange(self.start + self.t_buffer, self.end - self.T_seg + 10**-9, self.steps), 9) 
    
    def _segment_exclude_transition(self, t_patches, patches):
        """
        Parameter
        ---------
            patches: list
                Specify which patches to be used for segmentation
                (e.g.)
                [1, 2, 3, 4, 5] -> all patches
                [1, 3, 5] -> only no gap patches 
                [1, 2, 3, 4, 5, 'N'] -> include noise patch
        """
        #print('Segmentation without transition')
        t_arr = np.array([], dtype=float)
        
        for patch_idx in patches:
            # Start and end of the current patch
            # noise patch
            if patch_idx == 'N':
                # End points = global end points
                if self.end <= t_patches[-1]:
                    raise ValueError(
                        f'SegmentPointsProvider: global end point should be after {t_patches[-1]}s!' + 
                        'Please provide it using set_constants!'
                    )
                # Calculate back the starting point
                T_noise = self.T_seg + (self.N_patch_noise* self.N_seg - 1)* self.steps
                start = self.end - T_noise
                # Segmenting points in the noise part
                t_new = np.arange(start, self.end - self.T_seg + 10**-9, self.steps)
            # Weld patch
            else:
                start = t_patches[patch_idx-1]
                end = t_patches[patch_idx]
                t_new = np.arange(start + self.t_buffer, end - self.t_buffer  - self.T_seg + 10**-9, self.steps)
            
            # Append the segmentation points of the current patch
            t_arr = np.concatenate((t_arr, t_new))
        
        return np.around(t_arr, 9)
    
    
#========================================================================================
#========================================================================================
class SegmentAnnotator():
    """
    Comment on 2023/11/21(Tue): 
        This class is already adjusted to the latest measurement campaign iHub 2023. 
        Further modification is not required. 
    
    Example usage
    -------------
        annotator = SegmentAnnotator()
        
        # Specify the trial 
        trialNo = '22_0_3_Spalt_pneu'
        df_row = versuche[versuche['DEWETRON'] == trialNo]
        annotator.extract_gap_info(df_row)

        # Annotating each segment
        y_true = []
        for t in seg_points:
            (patch_idx, ratio) = annotator.compute_patch_number(t, t+T_seg, t_patches)
            gap = annotator.compute_gap(patch_idx, ratio) #[mm]
            classID, class_name = annotator.identify_class(gap, binary=False) # -> multi-class (binary==False)
            # Update
            y_true.append(classID)
        
        # Properly formulate the labels
        y_true = np.array(y_true).astype(int)
        class_bins = annotator.classes
    """
    
    def __init__(self):
        # global attributes
        self.gaps = None
    
    # ----------------------------
    # Remains constant for a single trial
    # ----------------------------
    def extract_gap_info(self, _df_row):
        """
        A single trial consistes of 5 + pre- and post-weld patches:
            signal = [pre-weld, P1, P2, P3, P4, P5, post-weld].

        The corresponding welding phases are following:
            on-weld (= no gap): P1, P3, P5
            off-weld (= gap): P2, P4

        The transition points associated to each patch are:
            pre-weld = 0 ... t_start (of welding)
            P1 = t_start ... t1
            P2 = t1 ... t2
            P3 = t2 ... t3
            P4 = t3 ... t4
            P5 = t4 ... t_end (of welding)
            post-weld = t_end ... T_meas

        """
        # Off-weld patches (= deliberately introduced gaps)
        P2 = float(_df_row['Spalt [mm]'].iloc[0]) # in [mm]
        P4 = P2
        # Nullspalt = On-weld patches (varying depending on the trial)
        # --> Comments on 23/11/21: for the iHub 2023 measurements, we don't have the 
        #     info on the gap size in the Nullspalte -> assign 0.0mm
        P1 = 0.0 # in [mm]
        P3 = 0.0 # in [mm]
        P5 = 0.0 # in [mm]
        self.gaps = tuple((P1, P2, P3, P4, P5))
        
    
    #=========================================================#
    # Segment variables
    #=========================================================#
    
    # ----------------------------
    # Determine on which patch the given time point is
    # ----------------------------
    def which_patch(self, t, _t_patch):
        """
        Parameters
        ----------
            t: float in [s]
                Time point to assess 
            _t_transition: np.ndarray(8) in [s]
                Set of weldgin phase transition points.
                It consists of 6 points: [t_start, t1, t2, t3, t4, t_end].

        Return
        ------
            patch_idx: int
                Indicate on which patch the given t is 
                (e.g.)
                    t_start < t < t1
                    => t is on P1 = t_start ... t1 
                    => patch_idx = 0

        """ 
        # Compute the nearest transition points in ascending order
        diff = np.around(np.abs(_t_patch - t), 10) # [s]
        diff_argsort = np.argsort(diff) # index

        # Nearest time point
        nearest = _t_patch[diff_argsort[0]]

        if t > nearest:
            # (e.g.) t1 < t < t2 => P2, and the nearest = t1
            patch_idx = diff_argsort[0]

        else:
            # (e.g.) t1 < t < t2 => P2, but the nearest = t2
            patch_idx = diff_argsort[0] - 1

        return patch_idx

    # ----------------------------
    # Function to compute the patch number 
    # ----------------------------
    def compute_patch_number(self, start, end, t_patches):
        """
        To compute the label, we need the following values:
            * patch: str
                = which patch the given segment is on
                -> in case the segment extends over two patches, the first patch = primary patch
            * ratio: float (0, 1]
                = how much of the segment is on the primary patch

        (e.g.) 
            (a) the complete semgent is on P2 -> label = (P2, 1.0)
            (b) only a part of the sevement is on P2 -> label = (P2, 0.4703)


        Parameters
        ----------
            start, end: float in [s]
                Start / end point of the segment to assess 
            t_patches: np.ndarray(8) in [s]
                Set of weldgin phase transition points.
                It consists of 8 points: [t_start, t1, t2, t3, t4, t_end].

        Return
        ------
            label: tuple(patch_idx, ratio)

        """ 
        # Determine the patch for the starting and the end points
        patch_start = self.which_patch(start, t_patches) # index
        patch_end = self.which_patch(end, t_patches) # index

        # Case (a) in the comment above
        if patch_start == patch_end:
            ## Label
            patch_idx = patch_start # index!!
            ratio = 1.0
        # Case (b) 
        else: 
            ## Label
            patch_idx = patch_start # index!!
            ## Calculate the ratio
            # Nearest transition point 
            nearest = t_patches[patch_end] #[s]
            # Calculate how much of the segment is on the first patch
            # start < nearest < end
            ratio = (nearest - start) / (end - start)
            # Avoid the floating point error of python
            ratio = round(ratio, 9)

        return (patch_idx, ratio)
    
    # ----------------------------
    # Compute the current gap
    # ----------------------------
    def compute_gap(self, patch_idx, ratio):
        # primary gap = gaps[patch_idx], secondary gap = gaps[patch_idx+1]
        if patch_idx < 4:
            gap = ratio* self.gaps[patch_idx] + (1 - ratio) * self.gaps[patch_idx+1] # [mm]
        # For the segments within the last patch 
        else:
            gap = ratio* self.gaps[patch_idx]
        # Eliminate the float point error
        return round(gap, 9)
        
    
    # ----------------------------
    # Label each segment
    # ----------------------------
    def identify_class(self, gap, binary, class_bins=None):
        # Binary classifier 
        if binary == True:
            return self.identify_class_binary_gap(gap)
        # IDMT classifier
        elif class_bins is None:
            return self.identify_class_multiclass(gap)
        # Gap width regression including the transition 
        else:
            return self.identify_class_regression(gap, class_bins)


    # ----------------------------
    # Labeling function for multi-class classification  
    # ----------------------------
    def identify_class_multiclass(self, gap):
        # ---------------------------------
        # Multi-class labeling 
        # -> classfying whether a segment corresnponds to no-gap or a 0.1mm gap, 0.2mm gap or 0.3mm gap. 
        # -> no-gap segments are considered to have negligiable sized gap
        # ---------------------------------
        # Only 4+1 classes 
        self.classes = ['noGap', 'Gap0.1', 'Gap0.2', 'Gap0.3', 'noise']
        
        # Determine the classID
        if np.isnan(gap) == True:
            class_name = 'noise' 
        elif gap < 0.1:
            class_name = 'noGap'
        elif gap == 0.1:
            class_name = 'Gap0.1'
        elif gap == 0.2:
            class_name = 'Gap0.2'
        else: # gap == 0.3
            class_name = 'Gap0.3'
            
        # Identify the class ID
        classID = self.classes.index(class_name)
        return classID, class_name
    
    # ----------------------------
    # Labeling function for binary classification (gap vs no gap)
    # ----------------------------
    def identify_class_binary_gap(self, gap):
        # ---------------------------------
        # here we exclude the trials with larger Nullspalte distance
        # -> classfying whether a segment corresnponds to no-gap or a gap 
        # ---------------------------------
        # Only 2+1 classes 
        self.classes = ['noGap', 'Gap', 'noise']
        # Determine the classID
        if np.isnan(gap) == True:
            class_name = 'noise'
        # no gap
        elif gap < 0.1:
            class_name = 'noGap'
        # gap
        else: 
            class_name = 'Gap'
        
        # Identify the class ID
        classID = self.classes.index(class_name)
        return classID, class_name
    
    
    # ----------------------------
    # Labeling function for regression tasks
    # ----------------------------
    def identify_class_regression(self, gap, _bins):
        """ 
        Identify the proper class label, if including the trials with a larger Nullspalte distnace. 
        The class label is determined as follows:
        (e.g.)
            _bins = [0.05, 0.1, 0.2, 0.3, 0.4, 10.0] # Edges!
            classes:
                0 = [0.0, 0.05) mm
                1 = [0.05, 0.1)mm
                2 = [0.1, 0.2)mm
                ....
            -> for the gap = 0.194 #[mm]
            -> label = bin index = 2 (-> representing 0.15mm <= gap < 0.2mm)

        """
        #------ Find the suitable classID
        diff_argsort = np.argsort(np.abs(_bins - gap))
        #print(diff_argsort)

        # Nearest edge value
        nearest = _bins[diff_argsort[0]] 

        if gap < nearest:
            # (e.g.) gap = 0.194 -> class = 2 = diff_argsort[0]
            classID = diff_argsort[0] 
        else:
            # (e.g.) gap = 0.206 -> class = 3 = diff_argsort[0] + 1
            classID = diff_argsort[0] + 1

        #------ Set the class_label
        if classID == 0:
            class_name = f'[0, {_bins[0]})mm'
        else:
            class_name = f'[{_bins[classID - 1]}, {_bins[classID]})mm'

        return classID, class_name

    
    
    
    
    
