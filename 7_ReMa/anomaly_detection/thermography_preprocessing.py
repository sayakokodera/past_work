import numpy as np
import cv2
import matplotlib.mlab as mlab
from thermography_file_reader import ThermoFileReader

class ThremoDataPreprocessing():
    def __init__(self):
        pass
        
    #=======================================
    #-------- Single image handling: data centering
    #=======================================
    
    def is_intarr(self, arr):
        if (np.array([0.2])).astype(arr.dtype) == 0:
            return True
        else:
            return False
    
    def adjust_dtype(self, image):
        # (a) correct format
        if image.dtype == 'uint8':
            return image
        # (b) need to adjust the data type
        else:
            return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    def threshold_otsu(self, image):
        # Adaptive thresholding = no need to manually selecting the threshold
        # Check the data type
        # (in theory, it can be float32, but for simplicity we convert the image into unit8)
        arr = self.adjust_dtype(image)
        # Otu's thresholding
        vmax = arr.max()
        ret,thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh
    
    # Centroid detection
    def compute_centroid(self, im_thresh):
        # calculate moments of binary image
        M = cv2.moments(im_thresh)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cy, cx)
        else:
            return None
    
    def translation_matrix(self, shift_x, shift_y):
        T = np.identity(3)
        T[0, -1] = shift_x
        T[1, -1] = shift_y
        return T
    
    def image_translation(self, img, shift_x, shift_y):
        # Reshape the input array, s.t. arr[:, :, i, j, ...] is vectorized
        #-> y-axis (height) changing the fastest
        Ny, Nx = img.shape[:2]
        if img.ndim == 2:
            img_vec = np.reshape(img, (Nx*Ny, 1), order='F')
        else: # in case the data array has more than 3 dimensions
            img_vec = np.reshape(img, (Nx*Ny, img.shape[-1]), order='F') 
        
        # (1) Original coordinates as a vector: y-axis chaging the fastest 
        xx_orig, yy_orig = np.meshgrid(np.arange(Nx), np.arange(Ny))
        yy_orig = yy_orig.flatten('F')
        xx_orig = xx_orig.flatten('F')
        # Stack the original coordinates 
        p_orig = np.array([xx_orig, yy_orig, np.ones(Nx*Ny)]).T
    
        # (2) Shifted coordinates
        T = self.translation_matrix(shift_x=shift_x, shift_y=shift_y)
        # Apply the transpose(!!) of the translation matrix 
        p_new = np.dot(p_orig, T.T)
        xx_new = p_new[:, 0].astype(int)
        yy_new = p_new[:, 1].astype(int)
    
        # (3) Allocate the pixel values according to the new coordinates
        img_new = np.zeros(img.shape, dtype=img.dtype)
        count = 0
        for x, y in zip(xx_new, yy_new):
            if x >= Nx:
                x = x % Nx
            if y >= Ny:
                y = y % Ny
            img_new[y, x, ...] = img_vec[count, ...]
            count += 1
    
        return img_new


    #=======================================
    #------ 3D data handling
    #=======================================
    def majority_vote(self, vec):
        unique, counts = np.unique(np.array(vec), return_counts=True)
        return unique[np.argmax(counts)]
        
    
    def centroid_majority_vote(self, _data_3D):
        # (1) centroid computing for every slie (= image)
        cy_all = []
        cx_all = []
        for this_frame in range(_data_3D.shape[-1]):
            this_im = _data_3D[..., this_frame]
            cy, cx = self.compute_centroid(self.threshold_otsu(this_im))
            cy_all.append(cy)
            cx_all.append(cx)
            
        # Find the majority vote centroid
        cx_maj = self.majority_vote(np.array(cx_all))
        cy_maj = self.majority_vote(np.array(cy_all))
        return (cy_maj, cx_maj)
        
        
    def centering(self, _data_3D, ret_shiftx=False, *args, **kwargs):
        # (1) Identify the shift in the x direction
        cy, cx = self.centroid_majority_vote(_data_3D)
        shift_x = int(_data_3D.shape[1]/2) - cx
        # (2) Correct the shift (we ignore the y-shift here)
        data_centered = self.image_translation(_data_3D, shift_x, 0)
        if ret_shiftx == True:
            return data_centered, shift_x
        else:
            return data_centered

    #=======================================
    #------ Freq. domain reconstruction
    #=======================================
    def reco_fft_phase(self, _data_time, *args, **kwargs):
        # Shape
        Ny, Nx, Nt = _data_time.shape
        # Identify the number of freq. bins
        freqs = np.fft.rfftfreq(Nt)
        Nf = len(freqs)
        # Instantiate
        Phase = np.zeros((Ny, Nx, Nf))

        for x in range(Nx):
            for y in range(Ny):
                this_sig = _data_time[y, x, :]
                this_phase, _ = mlab.phase_spectrum(x=this_sig)
                # Assign
                Phase[y, x, :] = this_phase
        return Phase

    #=======================================
    #------ Separating the shaft and the thred from a single image (transition image)
    #=======================================
    def separate_shaft_thred(self, arr):
        # Find the absmax between 158 & 170 --> where the contour toward the thred starts
        # start, end = 158, 170
        # boundary_y = start + np.argmax(np.abs(arr[start:end, :]), axis=0)
        start, end = 130, 150
        #boundary_y = start + np.argmin(np.abs(arr[start:end, :]), axis=0)
        # Or majority vote
        unique_elements, counts = np.unique(start + np.argmin(np.abs(arr[start:end, :]), axis=0), return_counts=True)
        boundary_y = unique_elements[np.argmax(counts)]* np.ones(arr.shape[1], dtype=int)
        # Separate
        shaft = np.copy(arr)
        thred = np.copy(arr)
        for col in range(arr.shape[1]):
            shaft[boundary_y[col]:, col] = np.nan
            thred[:boundary_y[col], col] = np.nan
        return shaft, thred
        
    
    #=======================================
    #------ Correcting the surface distortion
    #=======================================
    @staticmethod
    def surface_projection(angle_start, angle_end, length_pi=200):
        """
        Parameters
        ----------
            angle_start, angle_end: float in degree!! (not in rad)
                !!! important !!!
                    * both angle_start and angle_end needs to be between 0 and 180 degree
                    * angle_start < angle_end
                    -> then we can ensure that x_start is ALWAYS larger than x_end
            length_pi: int (in pixels)
                Length (or number of pixels) corresponds to 0 to 180 degree.
        """
        x_start = np.cos(np.deg2rad(angle_start))
        x_end = np.cos(np.deg2rad(angle_end))
        length = int(0.5* length_pi* (x_start - x_end))
        return length



#=======================================
# Illumination correction class
# -> here we select a threshold adaptively -> used for SymLogNorm
#=======================================
class ThermoIlluminationCorrection():
    def __init__(self):
        pass

    def set_parameters(self, p_start, num_bins, window_length, stepsize, method='median_unbiased'):
        """
        Parameters
        ----------
            window_length, stepsize: int
                Window length and stepsize for linear least square fit 
            p_start: float
                The lowest percentile where we want to start checking the slope/gradient
        """
        self.p_start = p_start
        self.num_bins = num_bins
        self.window_length = window_length
        self.stepsize = stepsize
        self.method = method

    def adaptive_threshold(self, arr):
        # (1) Compute the percentiles
        p = np.linspace(self.p_start, 100.0, self.num_bins) # percentages
        val = np.nanpercentile(np.abs(arr), p, method=self.method)
        # (2) Knee-point detection
        target, gradient, p_new = self.cdf_gradient(p, val/val.max())
        # Knee point = point where the gradient value first exceeds the target
        if gradient.max() > target:
            idx_knee = np.where(gradient > target)[0][0]
        else: # in case the steep increase at the last bin -> knee-point = last bin
            idx_knee = -2
        # (3) Threshold
        thres = val[idx_knee]
        return thres


    def ls_line_fit(self, x, y):
        # y = ax + b --> y = X \cdot v, with the coefficients v = [a, b]
        # (1) Zero-mean
        y_cent = y - np.mean(y)
        # (2) Array formatting
        X = np.stack((x, np.ones(x.shape[0]))).T
        # (3) Estimate the coefficients v via LS (pseudo-inverse): v_hat = X_pinv \cdot y
        [a_hat, b_hat] = np.dot(np.linalg.pinv(X), y_cent)
        return a_hat, b_hat+np.mean(y)

    def cdf_gradient(self, p, val):
        """
        Parameters
        ----------
            p: array, length = N
                Containing the percentiles where the cdf is computed
            val: array, length = N
                Values for each percentile
        """
        # (0) Setup
        N = len(p)
        # We need to make sure that we compute line-fit for the desired position, i.e.
        # the current p is in the middle of the window
        # -> shift the starting point! -> construct a p_new as the input to line-fit
        dp = p[1] - p[0]
        p_pre = np.flip(p[0] - dp* np.arange(1, int(self.window_length/2 + 1)))
        p_new = np.concatenate((p_pre, p))
        # (1) Determine the target slope
        slope_target = (val[-1] - val[0])/(p[-1] - p[0])
        # (2) Gradient estimation via sliding-window LS line fit
        xs = []
        gradient = []
        for start in range(0, N-self.window_length+1, self.stepsize):
            # Use p_new to make sure that the desired p is in the middle of the window
            x = p_new[start:start+self.window_length]
            y = val[start:start+self.window_length]
            a_hat, _ = self.ls_line_fit(x, y)
            # Append
            gradient.append(a_hat)
            xs.append(p[start])
        return slope_target, np.array(gradient), np.array(xs)


#============================================================
# Data preparation class
# -> does: 
#     (1) file reading 
#     (2) processing: 
#         centering
#         -> trimming
#         -> FFT reco
#         -> zero-mean
#============================================================

class ThermoDataPreparation():
    """
    Example usage
    -------------
        # Instantiate the data prep class
        prepper = ThermoDataPreparation()
        prepper.reader = path_rel_1
        # Load & process the data
        pos_y, pos_x = (-2, 3)
        reco = prepper.get_reco(pos=(pos_y, pos_x))
        fileNo = prepper.fileNo
    """
    
    def __init__(self):
        #====== Global image params ======
        # Known image paramters -> global params
        self.dtype = 'uint16'
        # Array formatting params: can be found in measurement.xml
        # Image size + frames
        self.width = 320
        self.height = 256
        # Number of time frames
        #frames = get_frames_from_xml(path_rel)#it is actually 75 frames, but few contain only 74 #250#int(data.shape[0]/(width*height)) 
        #self.ymin, self.ymax = 0, -1 : V250228, decided NOT to set them globally
        self.xmin, self.xmax = 25, -25
        
        # Post-centering: new segment shape 
        self.new_height = 209
        self.new_width = int(200* 90/180)#int(200* 90/180)
        self.new_shape = (self.new_height, self.new_width)
        #===================================

        # Instantiate the other global params
        self.fileNo = None 
        self.proc = ThremoDataPreprocessing()

    @property
    def reader(self):
        return self._reader
    @reader.setter
    def reader(self, _path_rel_1):
        self._reader = ThermoFileReader(_path_rel_1=_path_rel_1)
        

    def load(self, pos, *args, **kwargs):
        data_tens, self.fileNo = self.reader.load_data(
            pos=pos, 
            xmin=self.xmin, xmax=self.xmax, 
            dtype=self.dtype, width=self.width, height=self.height,
            ret_fileNo=True,
            **kwargs
        )
        return data_tens

    def process_time_domain(self, arr):
        # Center the data
        arr_centered = self.proc.centering(arr)
        # Remove the irrelevant part
        y_start = int(arr.shape[0]/2 - self.new_height/2)
        x_start = int(arr.shape[1]/2 - self.new_width/2)
        arr_trimmed = arr_centered[y_start:y_start+self.new_height, x_start:x_start+self.new_width]
        return arr_trimmed

    def compute_reco(self, arr_proc, *args, **kwargs):
        # FFT phase reco
        reco = self.proc.reco_fft_phase(arr_proc)[..., 1] # only fbin=1 is relevant
        # Zero-mean
        mean = reco.mean()
        reco = reco - mean
        return reco

    def get_processed_data_time(self, pos, *args, **kwargs):
        # (1) Load
        data_tens = self.load(pos, **kwargs)
        # (2) Process
        data_proc = self.process_time_domain(data_tens)
        return data_proc

    def get_reco(self, pos, *args, **kwargs):
        # (1) Load + time domain processing
        data_proc = self.get_processed_data_time(pos, **kwargs)
        # (2) Process + FFT reco
        reco = self.compute_reco(data_proc)
        return reco

    def get_fileNo(self, pos=None):
        if pos is not None:
            return self.reader.get_fileNo(pos_y=pos[0], pos_x=pos[1])
        else:
            return self.fileNo
        