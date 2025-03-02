import numpy as np

"""
Required parameters and attributes:
    * seg_shape: shape of each segment
        (# rows, # cols, ....)
    * ovl_height: height of the overlap (in pixels)
    * ovl_width: width of the overlap (in pixels)
    * numsegs_per_col: number of segments per column
    * numsegs_per_row: number of segments per row 
    * Size of the fidelity region = height and width (in pixels)
        (!!!) * fidelity region is considered to be in the center
              * NO(!!) overlap is allowed for fidelity region 
                 => only ONE segment is "active" in the fidelity region of a segment 
    * data type of the output 

Example usage:
    # (1) setup 
    stitcher = DataStitcher()
    stitcher.set_parameters(
        seg_shape=seg_shape, 
        numsegs_per_col=numsegs_per_col, 
        numsegs_per_row=numsegs_per_row
    )
    stitcher.segment_files = which_files
    stitcher.fidelity_region= (fid_height, fid_width)

    # (2) stitching
    output_arr = stitcher.stitch(
        func_loader=reader.load_data, 
        func_proc=thermo_preprocessing, 
        trim=False,
        normalize=False,
        weighting=True,
        y_offset=y_offset,
        xmin=xmin,
        xmax=xmax,
        new_shape=seg_shape[:2],
        screw_part=screw_part, 
        transition_mask=reader.transition_mask
    )
    # (3) Compute the fidelity_map masking + trimming --> visualization purpose 
    fid_map = stitcher.get_fidelity_map(ndim=output_im.ndim, dtype=output_im.dtype)
    im_masked = fid_map* output_im
    # Trimming
    im_trimmed = stitcher.trim(output_im)

"""

class DataStitcher():
    def __init__(self):
        # Initialize
        self._segment_files = None
        self._fidelity_region = None
        self.fidelity_map = None
        self.ovl_map = None
        # Step size to assmble the segmented data
        self.stepsize_height = None
        self.stepsize_width = None
        # Parameters relevant for overlap handling with Gaussian weights 
        self.sigma_x = 0.5
        self.sigma_y = 0.5
        self.mu1 = -0.5 # 1st peak of the Gaussian weight --> [To-Do] change it to the "edge" of the overlap
        self.mu2 = 0.5 # 2nd peak for the Gaussian weight


    def set_parameters(
        self, 
        seg_shape, 
        numsegs_per_col, numsegs_per_row
    ):
        """
        Parameters
        ----------
            seg_shape: tuple of int
                The shape of each segment in pixels ---> (height, width, ....) 
                    Height of each segment ---> corresponds to y-axis
                    Width of each segment ---> corresponds to x-axis
            Removed!!! : ovl_height, ovl_width: int 
                Number of PIXELS(!!!) that overlap between the consecutive images, in each direction, SINGLE-sided
                ---> this is used ONLY for calculating the step size!!! 
                ---> That's why removed!!!
            numsegs_per_col, numsegs_per_row: int
                Number of segments per column and per row 
                    per column --> # of y-axis segments --> related to height
                    per row --> # of x-axis segments --> related to width
                These parameters need to be provided, otherwise recovering them becomes combinatorial
        """
        self.seg_shape = seg_shape
        self.seg_height = seg_shape[0]
        self.seg_width = seg_shape[1]
        self.numsegs_per_col = numsegs_per_col
        self.numsegs_per_row = numsegs_per_row

    # =====================================
    # Handling of each segmented data 
    # =====================================
    @property
    def segment_files(self):
        """
        List of files (= each segmented data) to be loaded 
        """
        return self._segment_files

    @segment_files.setter
    def segment_files(self, arr_files):
        """
        Parameters
        ----------
            arr_files: str array (self.numsegs_per_row, self.numsegs_per_col)
                File names formated as an array of the same "size" as the resulting stitched image
        """
        # Sanity check
        if arr_files.shape != (self.numsegs_per_row, self.numsegs_per_col):
            raise ValueError(f'Stitcher: segment_files needs to be of the shape ({self.numsegs_per_row}, {self.numsegs_per_col})')
        self._segment_files = arr_files
        
    def load(self, fname):
        pass

    # =====================================
    # Overlap handling
    # =====================================
    def compute_stepsize(self, segsize, fidsize):
        r""" Compute the stepsize based on the size of the fidelity region
        ---> stepsize is used to assmble the segmented data
        """
        # Make sure that the stepsize becomes integer 
        # -> if not, reduce the fidelity size by 1
        if fidsize == 0:
            pass
        elif (segsize+fidsize)%2 != 0:
            fidsize -= 1
        stepsize = int((segsize+fidsize)/2)
        return fidsize, stepsize

            
    @property
    def fidelity_region(self):
        """
        Region within a SINGLE segment (!!!) where the accuracy is high (in pixel)
        -> useful to determine the weights for the overlaps with the neighboring segments
        >>> values:
            1.0 = for the area with high fideliy
            np.nan = the rest (i.e. low fidelity)
        """
        return self._fidelity_region

    @fidelity_region.setter
    def fidelity_region(self, size):
        """
        Prerequisite
        ------------
            * dimensions need to be selected, such that there will be NO overlap 
              of the fidelity regions in the fidelity map
              
        Parameters
        ----------
            size: tuple containing the follwoing two variables
                size = (region_height, region_width)
                >> region_height, region_width: int
                    "Height" / "width" of the area within a SINGLE segment, 
                    where the data fidelity is high

        Output
        ------
            _fidelity_region: np.array(self.seg_height, self.seg_width)
                This array consists of two parts
                    * center part, size = (region_height, region_width)
                        ---> here 1.0 is assigned (represent the are of a segment with high fidelity)
                    * the rest
                        ---> np.nan is assigned 
        """
        # Setting: as we use the fidelity region to determine the stepsize
        # ---> Make sure that the stepsize becomes integer
        self.fid_height, self.stepsize_height = self.compute_stepsize(self.seg_height, size[0])
        self.fid_width, self.stepsize_width = self.compute_stepsize(self.seg_width, size[1])
        # Center region where 1.0 is assigned
        row_start = int(0.5*self.seg_height - 0.5*self.fid_height)
        row_end = row_start + self.fid_height
        col_start = int(0.5*self.seg_width - 0.5*self.fid_width)
        col_end = col_start + self.fid_width
        
        self._fidelity_region = np.nan* np.ones((self.seg_height, self.seg_width))
        self._fidelity_region[row_start:row_end, col_start:col_end] = 1.0

    
    def compute_fidelity_map(self):
        """ 
        Compute a map/mask to indicate where the fidelty region of each segment is, 
        in the entire image. 
        --> self.fidelty_map is of the shape (self.height, self.width)
        !!! NaN in the self.fidelty_map needs to be converted to zeros first, 
         ===> otherwise NaN + 1.0 = NaN ===> the regions of ones cannot be overwritten by ones 
        """
        self.fidelity_map = self.stitch(
            func_loader=None, 
            func_proc=None,
            use_segment_shape=False,
            trim=False,
            input_arr=np.nan_to_num(self.fidelity_region),  
        )
        # # Corner handling -> corners will be replaced with 1s
        # corner_horizontal = int((self.seg_height - self.fid_height)/2)
        # corner_vertical = int((self.seg_width - self.fid_width)/2)
        # print(f'Corner handling: ({corner_horizontal}, {corner_vertical})')
        # yy, xx = np.meshgrid(np.arange(-corner_horizontal, corner_horizontal), np.arange(-corner_vertical, corner_vertical))
        # self.fidelity_map[yy, :] = 1.0
        # self.fidelity_map[:, xx] = 1.0
        # Convert zeros into NaN
        self.fidelity_map[self.fidelity_map==0] = np.nan

    def get_fidelity_map(self, ndim, dtype):
        return self.expand_to_n_dim(self.fidelity_map, n=ndim, dtype=dtype)  

    def compute_ovl_map(self):
        """
        A complev-valued map to illustrate where the overlaps are. 
        This map is required to properly compute the weights for the corner segments in
        the self.weights.
        >>> 0+0j, for the area with NO overlap
        >>> 1, for the area with y-dir. overlap
        >>> 1j, for the are with x-dir overlap
        >>> 1+1j, for the area with both y- and x-dir. overlap
        """
        # Parameters
        ovl_height = int(self.seg_height - self.stepsize_height)
        ovl_width = int(self.seg_width - self.stepsize_width)
        # Initialize 
        output_shape, startps_x, startps_y = self.stitching_metadata(use_segment_shape=False)
        self.ovl_map = np.zeros(output_shape, dtype=complex)
        # Assign True -> using np.add.outer to avoid two for loops
        for y in startps_y[1:]:
            self.ovl_map[y:y+ovl_height, :] += 1.0
        for x in startps_x[1:]:
            self.ovl_map[:, x:x+ovl_width] += 1j
        
    def compute_mask(self, _map, pos):
        """
        Computing a mask for a SINGLE segment, where a segment for the desired position is
        extracted from the given map. 
        
        Parameters
        ----------
            pos: np.array(row, col), both are int
                Position of the considered segment. 
                (!!!) this is given in terms of the segments, NOT in pixels! 
                (e.g.) for the segment located in row = 2, col = 3 ===> pos =(2, 3) 
        """
        # (1) Start and end indices
        row, col = pos
        x_start = col* self.stepsize_width
        x_end = x_start + self.seg_width
        y_start = row* self.stepsize_height
        y_end = y_start + self.seg_height
        # Extract
        return _map[y_start:y_end, x_start:x_end]
        
    def nan_mask(self, pos):
        """
        A mask for a SINGLE segment, where the NaN mask is computed depending on the 
        segment position from the global fidelity_map.
        
        Parameters
        ----------
            pos: np.array(row, col), both are int
                Position of the considered segment. 
                (!!!) this is given in terms of the segments, NOT in pixels! 
                (e.g.) for the segment located in row = 2, col = 3 ===> pos =(2, 3) 
        """
        # check if self.fidelity_map is already instantiated 
        if self.fidelity_map is None:
            self.compute_fidelity_map()
        return self.compute_mask(self.fidelity_map, pos)


    def zero_mask(self, pos):
        """
        A mask for a SINGLE segment to avoid overwriting the fidelity region of the 
        neighboring segments.
        --> area where the neighbor segments cover as their fidelity region will be set 0.0
        in the considered segment.
         
        Parameters
        ----------
            pos: np.array(row, col), both are int
                Position of the considered segment. 
                (!!!) this is given in terms of the segments, NOT in pixels! 
                (e.g.) for the segment located in row = 2, col = 3 ===> pos =(2, 3) 

        Remark
        ------
            If the current nan_mask is identical to the fidelity region, then the current 
            zero_mask should be np.zeros((seg_height, seg_width)).
        """
        nan_mask = self.nan_mask(pos)
        # convert the nan mask to zero mask
        nonzeros = (np.nan_to_num(nan_mask) - np.nan_to_num(self.fidelity_region)).astype(bool)
        return ~nonzeros # boolean array

    def ovl_mask(self, pos):
        """
        A mask for a SINGLE segment, where the OVL mask is computed depending on the 
        segment position from the global ovl_map.
        
        Parameters
        ----------
            pos: np.array(row, col), both are int
                Position of the considered segment. 
                (!!!) this is given in terms of the segments, NOT in pixels! 
                (e.g.) for the segment located in row = 2, col = 3 ===> pos =(2, 3) 
        """
        # check if self.fidelity_map is already instantiated 
        if self.ovl_map is None:
            self.compute_ovl_map()
        return self.compute_mask(self.ovl_map, pos)

    def expand_to_n_dim(self, X, n, dtype=None):
        """
        Adjust the dimension of the fidelity region / map or the output array (for self._stitch). 
        
        Parameters
        ----------
            X: np.array(n1, n2) 
                An input matrix to be expanded 
            n: int
                The desired dimension 
            
        """
        while X.ndim < n:
            X = np.expand_dims(X, axis=-1)  # Add a new axis at the end

        if dtype is not None:
            X = X.astype(dtype)
        return X

    def trim(self, arr):
        """
        Trim the stitched data based on the fidelity_map
        -> the areas where the fidelity_map is NaN are removed from the stitched data

        Parsmeters
        ----------
            arr: np.ndarray
                Stitched data
                (!!) needs to be with the same size as self.fidelity_map
        """
        # (1) Non-NaN mask
        non_nan_mask = ~np.isnan(self.fidelity_map)
        # Find the indices of rows and columns that have at least one non-NaN value
        valid_rows = np.any(non_nan_mask, axis=1)
        valid_cols = np.any(non_nan_mask, axis=0)
        # Trim
        arr_trimmed = arr[valid_rows, ...][:, valid_cols, ...]
        return arr_trimmed

    def weights(self, pos):
        """
        Thoughts:
        ---------
            * Weights neeed to be computed depending on the area of overlap
            ---> should be adjusted according to the data!
            * Some blur weighting would be probably more appropriate than simple uniform weighting
            * I should compute the weight on the ENTIRE image / data region 
              ---> to avoid considering the positions etc 
              ---> for that fidelity_map can be used 

        Steps
        -----
            (0) Compute the overlap map using self.compute_ovl_map()
            (1) Compute the blending weights (linear, Gaussian etc)
            (2) Swap the center part with the current nan_mask, i.e. the current(!!!) fidelity region
                ---> (a) "gray zone" wilkl be taken care of by self.zero_mask
                    ===> so no further componentds required)
                    * gray zones = the region which are NOT a part of the fidelty region of any segments, i.e. 
                                   left out as non-fidelty areas
                ---> (b) it is NOT the global fidelity region!!! (v241223)
                    ---> because the corners have different fidelity region than the center segments
                    ===> use nan_mask to identify the current fidelity region
                    (because nan_mask is computed based on the fidelity_map and the current segment position)
        """
        # (0) OVL
        ovl_mask = self.ovl_mask(pos)
        # (1) Compute the weights for blending (currently: linear blending, V241223)
        # First, we compute the weights for y and x direction separately
        # (so that we can properly adjust the weights for the corner segments) 
        Wy, Wx = self.linear_weights(self.seg_shape[:2]) 
        # (2) Reflect the directon-dependent OVL of the current(!!) segment
        # --> assign 1.0 to the weight matrix of each direction, if there is no overlap
        Wy[np.array(ovl_mask.real < 1.0)] = 1.0
        Wx[np.array(ovl_mask.imag < 1.0)] = 1.0
        W = Wx* Wy
        return W

    def trapez_vector(self, peak, N):
        vec1 = np.linspace(0.0, 1.0, peak)
        vec2 = vec1[::-1]
        vec = np.ones(N)
        vec[:peak] = vec1
        vec[-peak:] = vec2
        return vec

    def linear_weights(self, shape):
        # This function computes linear weights that peaks at the edges of the fidelity region
        # and decays linearly toward the segment edge
        Ny, Nx = shape
        # Create two linear increasing / decreasing lines 
        #---- x direction 
        x_peak = int(0.5*(self.seg_width - self.fid_width))
        vec_x = self.trapez_vector(x_peak, Nx)
        Wx = np.tile(vec_x, (Ny, 1))
        #---- y direction 
        y_peak = int(0.5*(self.seg_height - self.fid_height))
        vec_y = self.trapez_vector(y_peak, Ny)
        Wy = np.tile(vec_y, (Nx, 1)).T
        # Resulting weights = Hadamard product 
        #W = Wx* Wy
        return Wy, Wx
        
    
    def Gaussian_weights_double(self, shape):
        # !!! Currently not working properlly!! 
        # This function generates a Gaussian weights that has two peaks along the x-axis
        # Some gobal parameters are set in __init__
        Ny, Nx = shape
        # Create meshgrid representing pixel coordinates
        y = np.linspace(-1, 1, Ny)
        x = np.linspace(-1, 1, Nx)
        xx, yy = np.meshgrid(x, y)
        # First Gaussian centered at mean1_x
        W1 = np.exp(-0.5* (((xx - self.mu1)/self.sigma_x)**2 + (yy/self.sigma_y)** 2))
        # Second Gaussian centered at mean2_x
        W2 = np.exp(-0.5* (((xx - self.mu2)/self.sigma_x)**2 + (yy/self.sigma_y)** 2))
        # Sum up
        W = W1 + W2
        # Normalize
        W = W / W.max()
        return W

    # =====================================
    # Processing of each segment
    # =====================================
    def preprocessing(self, arr, pos, func_proc=None, weighting=False, **kwargs):
        """
        Processing that needs to be applied to an individual (!!!) segment 
        (i.e. not a group action)
        
        Parameters
        ----------
            arr: np.array 
                A single segment data 
            pos: array(row, col)
                Position of the selected segment (NOT in the pixels)
        """
        # (1)Apply data-specific preprocessing
        if func_proc is not None:
            arr_proc = func_proc(arr, pos=pos, **kwargs)
        # (2) Avoid assigning the values to the fidelity region of the neighbor segments
        row, col = pos
        zero_mask = self.zero_mask(pos)
        arr_proc = self.expand_to_n_dim(zero_mask, n=arr_proc.ndim)* arr_proc
        arr_proc = (arr_proc).astype(arr_proc.dtype)
        # Optional: weight the overlap area
        if weighting == True:
            # (3) Weights --> accounting for the overlap areas
            W = self.weights(pos)
            arr_proc = self.expand_to_n_dim(W, n=arr_proc.ndim)* arr_proc
        return arr_proc
        
            
    # =====================================
    # Stitching
    # =====================================       
    def stitching_metadata(self, use_segment_shape):
        """
        Compute the meta information for stitching to which includes
            * shape of the output array
                * height of the output array
                * width of the output array
                * (optional) the rest of the dimension adjusted to the segments
            * ALL starting points for stitching along width, i.e. x
                -> starting points to assign each segment 
            * All starting points for stitching along height, i.e. y. 
        Why do we have this? It is because
            * I don't want to run for loops more than once 
            * having the metadata makes it easier to track back what's happening (-> easier to find errors)

        Parameter
        ---------
            use_segment_shape: boolean
                True, if the rest of the dimension needs to be identical to each segment 
                (False, if the fidelity map is computed)

        Returns
        -------
            output_shape: int array
                (height, width, .....)
            startps_x, startps_y: int vectors 
        """
        # << note >> we need to take the overlap into account 
        # y (= height or row)
        height = (self.numsegs_per_row - 1) * self.stepsize_height + self.seg_height
        startps_y = np.arange(0, self.numsegs_per_row)* self.stepsize_height
        # x (= width or col)
        width = (self.numsegs_per_col - 1) * self.stepsize_width + self.seg_width
        startps_x = np.arange(0, self.numsegs_per_col)* self.stepsize_width

        # Adjusting the rest of the dimension / shape 
        if use_segment_shape == False:
            output_shape = (height, width)
        else:
            output_shape = (height, width) + self.seg_shape[2:]
            
        return output_shape, startps_x, startps_y 

    
    def stitch(self, func_loader, func_proc, 
               use_segment_shape=True, dtype=None, trim=False, input_arr=None, 
               normalize=False, weighting=False,
               *args, **kwargs):
        """
        Prerequisite
        ------------
            * self.set_parameters needs to be computed 
            * self.segment_files needs to be set  
            
        Parameters
        ---------- 
            use_segment_shape: boolean (True by default)
                True, if the output array needs to have the same dimension as the segments
                (False, for computing the fidelity map)
            dtype: str (None by default ---> then automatically becomes float64)
                Data type to specify, if necessary 
            trim: Boolean (False by dafault)
                True, if the segments need to be trimmed 
            input_arr: array, None by default
                This input_arr is required to compute the fidelity map. With that said,
                it is not required, if we want to load the segmented data. 
                ---> input_arr needs to be specified, if func_loader is None
            weighting: boolean (False by default)
                True, if the overlap area needs to be weighted

        """
        # Compute the metadata
        output_shape, startps_x, startps_y = self.stitching_metadata(use_segment_shape)
        #print(f'output_shape={output_shape}, startps_x={startps_x}, startps_y={startps_y}')
        # Initialize 
        output_arr = np.zeros(output_shape, dtype=dtype)
        
        # Loop through the segment files and place them in the stitched array
        for col, x in enumerate(startps_x):
            for row, y in enumerate(startps_y):
                # (1) Load
                # case 1: compute the fidelity map 
                if func_loader is None:
                    if input_arr is None:
                        raise AttributeError('DataStitcher._stitch: input_arr cannot be None, if func_loader is None') 
                    segment = input_arr
                # Case 2: actual sticthing with the segmented data 
                else:
                    # Comment on 24/11/29: in case we want to raise a "transition flag" in the thermo.reader,
                    # we need to reset the flag here, just in case, i.e. reader.transition_flag = None or so
                    this_segment = func_loader(self.segment_files[row, col], **kwargs)
                    # (2) data preprocessing: indvidual actions (e.g. zero-masking)
                    segment = self.preprocessing(
                        arr=this_segment, pos=(row, col), 
                        func_proc=func_proc, weighting=weighting,
                        **kwargs
                    )
                    # Adjust the segment size
                    if segment.ndim > 2: # 3+D case
                        # Adjust the segment size
                        segment = segment[..., self.seg_shape[2:]]
                    else: #2D case (e.g. FFT reco of the thermography data) -> we need to add another dimension of size 1
                        segment = segment[..., np.newaxis]
                    if normalize == True:
                        # Vmax for normalization 
                        vmaxs = np.max(np.abs(segment), axis=(0, 1))
                        segment = segment / vmaxs

                # (3) Stitch
                output_arr[y:y + segment.shape[0], x:x + segment.shape[1], ...] += segment

        # Post-processing: group actions (e.g. weighting, trimming etc)
        if trim == True:
            output_arr = self.trim(output_arr)

        return output_arr 
