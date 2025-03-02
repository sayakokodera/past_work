import numpy as np
import pandas as pd


class DataSplitter():
    # ----- Segmpents selection
    # 2 important notes:
    # (1) Shorter duration segments tend to have much larger sample size than the longer segments. 
    # This causes an unfiar training for NCA and consequently makes the kNN accuracy unreliable.
    # -> we need to stick to the same number of the segments regardless of the segment duration!
    # -> select 1 segment / patch, exceot the noise patch. Noise patch seems to require 3 segments.
    # (2) Sample size may significantly vary with classes for multi-class clustering
    # This causes an imbalanced training/test 
    # -> we need to use different selection scheme depending on the clustering type
    
    """
    Example usage:
    ---------------
        splitter = DataSplitter()
        splitter.metadf = metadf
        
        ===== option 1 ====
        # --- simply splitting all segments in the metadata
        splitter.N_seg_class = 30
        id_train, id_test = splitter.data_split(ratio=0.3)
        
        ===== option 2 =====
        # --- Trial-wise train/test split
        trials_train, trials_test = splitter.trials_split(
            ratio=0.2, 
            ret_ID=False,
            trials2drop=['1_11', '1_8', '1_18', '2_16', '3_30'], 
            seed=10
        )
        id_train = splitter.select_segments_balanced_class(
            trials=trials_train, 
            N_seg_class = 30
        )

        ===== option 3 =====
        # --- Trial selection as a stand-alone
        # Test set
        trials_test = splitter.trials_selection(
            N=5, 
            trials2drop=['1_11', '1_8', '1_18', '2_16', '3_30'], 
            trials2include=['1_16', '2_22', '3_3'], 
            seed=5
        )
        id_test = splitter.get_dataIDs(trials_test)
        
    """
    
    def __init__(self):
        pass
    
    # ----------------------------
    # Parameters
    # ----------------------------
    @property
    def metadf(self):
        return self._metadf
    
    @metadf.setter
    def metadf(self, metadf):
        """
        Parameter
        ---------
            metadf: pandas dataframe
                metadata 
        """
        self._metadf = metadf.copy()
    
    @property
    def N_seg_class(self):
        return self._N_seg_class
    
    @N_seg_class.setter
    def N_seg_class(self, N_seg_class):
        """
        Parameter
        ---------
            N_seg_classes: int
                number of segments to be selected for each class
                
        """
        self._N_seg_class = N_seg_class
        
    # ----------------------------
    # Select 1 segment / patch
    # ----------------------------
    def patch_wise_selecion(self, trial_df, N=1, N_noise=3, shuffle=False, rng=None):
        """
        Main purpose = select 1 segment / patch
            * for a SINGLE trial use 
            * trial_df can contain arbitrary number of classes (all, one, combination...)
        """
        # Initialize the random generator
        if rng == None:
            rng = np.random.default_rng()
        # base
        idx_selected = np.array([], dtype=int)
        # iterate over patches
        for patch in trial_df['patch'].unique():
            # List up all dataIDs for the current patch
            indices = list(trial_df[trial_df['patch'] == patch]['dataID'])
            # select N indices
            if patch == 'N':
                selection = rng.choice(indices, size= N_noise, replace = False)
            else:
                selection = rng.choice(indices, size= N, replace = False)
            # Update
            idx_selected = np.concatenate((idx_selected, selection))
            
        if shuffle == False:
            return np.sort(idx_selected)
        else:
            return idx_selected
    
    
    # ----------------------------
    # Segment selection for keeping the 
    # data size equal for all classes
    # ----------------------------
    def select_segments_balanced_class(self, trials=None, N_seg_class=None, ret_arr=False, seed=None):
        """
        Parameters
        ----------
            trials: list/array
                Collection of trial numbers, s.t. the segments are selected only from these trials
            N_seg_class: int
                Number of segments for each class. This ensures a balanced training.
            ret_arr: boolean (false by default)
                True, if the selected dataIDs are to be returned as an array. 
                In this case, id_selected is reshaped s.t. each column contains the IDs of the 
                same class. 
                    (e.g.) id_selected[:, 0] = selected IDs for the class 0
            seef: int (None by default)
                Seed number for random generator
                
        Output
        ------
            id_selected: list 
                Collection of segment IDs. This list is "sorted" according to the class of
                each segment.
                => Segment IDs for the class 0 are placed at the beginning of this list. 
        """
        if N_seg_class != None:
            self.N_seg_class = N_seg_class
        if self.N_seg_class == None:
            raise AttributeError('DataSplitter: N_seg_class should be provided!')
            
        # Adjust the dataframe to the trials
        if type(trials) == type(None):
            df = self.metadf
        else:
            df = self.get_newdf(trials)
        
        # Instantiate 
        id_selected = np.array([], dtype=int)

        for curr_ID in sorted(list(df['classID'].unique())):            
            class_df = df[df['classID'] == curr_ID].reset_index()
            # Select dataIDs for the current classID
            new_segIDs = self.class_wise_selection(class_df, seed)
            # Update
            id_selected = np.concatenate((id_selected, new_segIDs))
            
        if ret_arr == False:
            return id_selected
        else:
            return id_selected.reshape((self.N_seg_class, curr_ID+1), order='F')
    
    
    def class_wise_selection(self, class_df, seed):
        """
        With this function, the same number of dataIDs are selected for a SINGLE class.
        This can be used to keep the sample size constant for all classes.
        
        Parameters
        ----------
            class_df: pandas dataframe
                segments information of a single CLASS as a pandas dataframe
                (i.e. it contains the information of only ONE class)
            seed: int
        Return
        ------
            idx_selected: array of indices, shape = self.N_seg_class
                selected dataIDs
                
        Comments
        --------
            this way of selecting the segments is obviously not efficient for noGap and noise classes, 
            as they are present in all trials. 
            This means that we'll iterate over the same trials mutiple times. 
            This issue will be solved in the future. 
        """
        # Instantiate the random generator
        rng = np.random.default_rng(seed=seed)
        
        # base
        id_selected = np.array([], dtype=int)
        for trialNo in class_df['trial'].unique():
            # Select one segment per patch
            curr_selection = self.patch_wise_selecion(
                class_df[class_df['trial'] == trialNo], 
                shuffle=True, 
                rng=rng,
            )
            # update
            id_selected = np.concatenate((id_selected, curr_selection)) 
        
        # Shuffle        
        return rng.permuted(id_selected)[:self.N_seg_class]
    
    
    # ----------------------------
    # Data split
    #   Splitting the segments into training and test set. 
    #   All trials are included here 
    #   = different segments from the same trials may be used for both training and test. 
    # ----------------------------
    def data_split(self, ratio, df=None, seed=None):
        """
        Parameters
        ----------
            ratio: float (0.0, 1.0]
                Ratio for the data size of the test set
                (e.g. ratio = 0.2 -> test : training = 2 : 8)
        """
        # Variable check
        if type(df) == type(None):
            df = self.metadf
        if self.N_seg_class == None:
            raise AttributeError('DataSplitter: class variable N_seg_class should be provided!')
            
        # Select IDs for both training & test: shape = N_seg_class x number of classes
        id_selected = self.select_segments_balanced_class(df, seed, ret_arr=True) 
        # Split
        id_test = id_selected[:int(ratio* self.N_seg_class), :].flatten('F')
        id_train = id_selected[int(ratio* self.N_seg_class):, :].flatten('F')
        return (id_train, id_test)
        
    
    
    # ----------------------------
    # Trials split
    # ----------------------------
    def trials_split(self, ratio, ret_ID=False, trials2drop=None, seed=None):
        """
        Split the trials into the training and the test set. 
        Note: ALL trials in the self.metadf are considered.
        
        Parameters
        ----------
            ratio: float (0.0, 1.0]
                Ratio to compute the number of trials for the TEST set
                (e.g. ratio = 0.2 -> test : training = 2 : 8)
            trials2drop: list/array (None by default)
                If some trials are to be excluded, they can be specified here.
                *** important note ***
                    In this class, these trials are removed from self.metadf.
            ret_ID: boolean (False by default)
                True, if all dataIDs of the selected trials are also to be returned
            seed: int (None by default)
                Seed number 
        """
        # Modify the metadf
        if type(trials2drop) != type(None):
            self.metadf = self.drop_trials(self.metadf, trials2drop)
            print('DataSplietter: metadf is modified and dropped some trials!')
        
        # Data size
        N = len(self.metadf.trial.unique())
        N_test = int(ratio* N)
        N_train = N - N_test
        # Test set
        trials_test = self.trials_selection(
            N=N_test, 
            seed=seed
        )
        # Training set: don't forget to exclude the test trials
        trials_train = self.trials_selection(
            N=N_train, 
            trials2drop=trials_test, 
            seed=seed
        )
        
        if ret_ID == False:
            return (trials_train, trials_test)
        else:
            id_train = self.get_dataIDs(trials_train)
            id_test = self.get_dataIDs(trials_test)
            return (trials_train, id_train), (trials_test, id_test)

    
    def trials_selection(self, N, trials2drop=None, trials2include=None, seed=None):
        """
        Parameters
        ----------
            N: int
                Number of trials to be selected
            trials2drop: list/array (None by default)
                If some trials are to be excluded, they can be specified here.
                Note that self.metadf remains uncahnged. 
            trials2include: list/array (None by default)
                If some trials are to be included, they can be spcified here.
            seed: int (None by default)
                Seed number 
        Return
        ------
            trials_selected: numpy array (object), shape = N
                Selected trials
                
        Comments:
        ---------
            * dataIDs can be easily obtained using get_dataIDs
            * corresponding dataframe can be extracted from the metadf using get_newdf
        """
        # Copy / base
        df = self.metadf.copy()
        trials_selected = np.array([], dtype=object) # -> for easily including the trials2include
        
        # Exclude the given trials
        if type(trials2drop) != type(None):
            df = self.drop_trials(df, trials2drop)
        if type(trials2include) != type(None):
            # Exclude trials2include from the df
            df = self.drop_trials(df, trials2include)
            # Include the specified trials in the selected trials
            trials_selected = np.array(trials2include)
            # Adjust the data size
            N = N - len(trials2include)
            # Update the dataframe
            trials_rest = self.drop_trials(df, trials2include).trial.unique()
        
        # Select the trials (not sorted!)
        trials_rest = df['trial'].unique()
        rng = np.random.default_rng(seed=seed)
        selection = rng.choice(trials_rest, size=N, replace=False)
        trials_selected = np.concatenate((trials_selected, selection)) # NOT sorted (e.g. '1_29' before '1_3')
        
        # Update the dataframe -> trials will be sorted properly
        df_selected = self.get_newdf(trials_selected)
        trials_selected = df_selected['trial'].unique()
        return trials_selected
        
        
    # ----------------------------
    # Dataframe handling
    # ----------------------------
    # Exclude specifict trials
    def drop_trials(self, df, trials2drop):
        idx2drop = np.array([], dtype=int)
        for elem in trials2drop:
            idx2drop = np.concatenate((idx2drop, df[df.trial == elem].index))
        new_df = df.drop(idx2drop, axis=0, inplace=False)
        return new_df.reset_index(drop=True)


    def get_dataIDs(self, trials):
        df = self.get_newdf(trials)
        return np.array(df.dataID).astype(int)
        
    
    def get_newdf(self, trials):
        """
        This function returns a new dataframe extracted from the metadf, which includes only
        the desired trials.
        
        Parameter
        ---------
            trials: list/array
                Desired trials to be included in the new dataframe
                If it is not in a list/array form, its data format is modified. 
        """
        # Modify the data format if necessary
        if type(trials) == str:
            trials = [trials]
        return self.metadf[self.metadf.trial.isin(trials)].reset_index(drop=True)

