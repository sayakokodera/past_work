##### ESE Analysis for image quality ########

import numpy as np
from scipy.stats import norm

        
########################################################################################################## MSE #########        
class ImageQualityAnalyzerMSE():
    # constructor
    def __init__(self, reference):
        """ Constructor        
        Parameters
        ----------
            reference : 2D array (np.array(Nx, Ny)), required for ImageQualityAnlyzer
                the reference data to be compared with

        """
        super().__init__() 
        self.reference = np.array(reference)
        self.Nx = self.reference.shape[0]
        self.Ny = self.reference.shape[1]
        self.alpha = None
        self.mse = None
        self.data_err = None
        
    def _check_dimension(self):
        if self.Nx == self.data_err.shape[0] and self.Ny == self.data_err.shape[1]:
            pass
        else :
            raise AttributeError('ImageQualityAnalizer : dimension of the input data and the reference do not match')
        
        
    def _calculate_mse(self):
        """ calculate the MSE with alpha for bettrer scaling s.t. 0 < MSE < 1  
        """
        ref = np.array(self.reference)
        data = np.array(self.data_err)
        # calculate the alpha
        ref_vec = ref.reshape(self.Nx* self.Ny, 1)
        data_vec = data.reshape(self.Nx* self.Ny, 1)
        self.alpha = float(np.dot(np.transpose(ref_vec), data_vec) / np.dot(np.transpose(data_vec), data_vec))
        # MSE calculation
        num = np.linalg.norm(self.alpha* data - ref)
        den = np.linalg.norm(ref)
        self.mse = num / den #se 
        #self.mse = np.mean(se)
        
    
    def get_mse(self, data_err):
        """ with this function the mean squared error (MSE) is calculated 
        Parameters
        ----------
            data_err : 2D array (np.array(Nx, Ny))
                the data which should be compared with the given reference data
                
        Returns
        -------
            mse : float
        """
        self.data_err = np.array(data_err)/np.max(np.abs(self.reference))  
        self.reference = self.reference / np.max(np.abs(self.reference))
        self._check_dimension()
        self._calculate_mse()
        return self.mse
    

######################################################################################################### GCNR #########
class ImageQualityAnalyzerGCNR():
    # constructor
    def __init__(self, data, roi = None):
        """ Constructor        
        Parameters
        ----------
            data : 2D array
                data to evaluate
            roi : 2D array of Nx and Ny (default is None)
                indicates the where our ROI is
                e.g. when (x, y) = (25, 36) is within the ROI, but (x, y) = (71, 89) is located outside of the ROI, 
                then roi[25, 36] = 1, but roi[71, 89] = 0

        """
        super().__init__()
        
        self.data = np.array(data) / np.abs(data).max() # normalized with its maximal
        self.Nx = self.data.shape[0]
        self.Ny = self.data.shape[1]
        self.Npixel = self.Nx* self.Ny
        if roi is None:
            self.roi = np.zeros((self.Nx, self.Ny))
        else:
            self.roi = roi
        self.epsilon = None
        self.pdf_inside = None
        self.pdf_outside = None
        self.p_inside = None # probability of the signal inside teh ROI
        self.p_outside = None # probability of the signal outside of the ROI
        self.p_miss = None # probability of the undetected pixels within the ROI
        self.p_false = None # probability of the falsely detected pixels
        self.p_err = None # error probability (-> used to find the optimal epsilon)
        self.pix_missed = None # number of the missed pixels (i.e. below self.epsilon inseide ROI)
        self.pix_false = None # number of the falsely detected pixels (i.e. over epsilon outside ROI)
        self.ovl = None
        self.gcnr = None
        self.width_outside = None
        self.m_gcnr = None
        
    
    # determine the ROI
    def set_roi(self, reference, pos_defect, threshold):
        """ this function will be required, when the ROI is not known.
        The ROI will be determined by using the defect position information and the given threshold.
        If the aplitude of the reference data at a particular position, say (x1, y1), is bigger than the 
        threshold* array[x_def, y_def], then (x1, y1) will be included into ROI.
        
        Parameters :
        ------------
            reference : 2D array
                to determine the ROI
            pos_deffect : 2D array with the size of [Ndefect, 2]
                containing the information of (x_def, y_def) for each defect 
            threshold : float (= 0...1)
                threshold for determining the ROI
        """
        ref = np.array(reference)
        
        #self.roi[x, curr_def[1]]
        
        
        for curr_def in pos_defect:
            roi_x = np.where(ref[:, curr_def[1]] >= threshold* max(ref[:, curr_def[1]]))[0]
            roi_y = np.where(ref[curr_def[0], :] >= threshold* max(ref[curr_def[0], :]))[0]
            # check the size of roi_x and roi_y
            if len(roi_x) != len(roi_y):
                raise AttributeError('ImageQualityAnalyzer : shape of the reference data should be symmetric!')
            # add them to the roi
            for x in roi_x:
                self.roi[x, curr_def[1]] = 1
            for y in roi_y:
                self.roi[curr_def[0], y] = 1
        
        
    # get the sorted values in the desired region
    def _sort_values(self, region):
        """
        Parameters : 
        ------------
            region : 2D array 
                containing the location info. on the elements inside the desired region
                e.g. : k-th element is (x1, y1) --> region[k, 0] = x1, region[k, 1] = y1
        
        Return :
        --------
            values_list : list
                sorted list, containing all values within the given region (which can be regarded as PDF)
        """
        values_list = []
        for x, y in zip(region[:, 0], region[:, 1]):
            values_list.append(self.data[x, y])
        # sort the list
        values_list.sort()
        return values_list


    # calculate PDF i.e. all values at each pixels located inside or oustside of the ROI
    def _calculate_pdf(self):
        """ this function provides the PDF of inside and outside of the ROI and their probabilities.
        
        Retruns :
        ---------
            None, but following attributes are newly calculated
            
            self.pdf_inside : a sorted list
                containing all values inside the ROI
            self.pdf_outside : a sorted list
                containing all values outside the ROI
            self.p_inside : float
                probability of the signal inside the ROI = pixels inside ROi / all pixels
            self.p_inside : float
                probability of the signal inside the ROI = pixels outside ROi / all pixels
                
        """
        ### inside the ROI ###
        # specify the region
        region_inside = np.array([np.where(self.roi == 1)[0], np.where(self.roi == 1)[1]]).transpose()
        # get PDF
        self.pdf_inside = self._sort_values(region_inside)
        # probability
        self.p_inside = len(self.pdf_inside) / self.Npixel
        
        ### outside the ROI ###
        # specify the region
        region_outside = np.array([np.where(self.roi == 0)[0], np.where(self.roi == 0)[1]]).transpose()
        # get PDF
        self.pdf_outside = self._sort_values(region_outside)
        # probability
        self.p_outside = len(self.pdf_outside) / self.Npixel
        


    # optimize the epsilon
    def get_minimum_overlap(self):
        """ this function yields the optimal epsilon which minimize the error probability.
        Epsilon will be picked from the values available in the data to avoid unnecesarry calculations.
        
        Returns :
        ---------
            None, however following attributes are newly set :
                
            self.epsilon : float
                values chosen form the self.data, which minimizes the p_err
            self.p_err : float
                error probability, i.e.e the sum of the probability of the missed detections and the false detections
            self.p_miss : float
                the probability of teh missed detections
            self.p_false :
                the probability of the false detections
                
        """
        
        values_inside_roi = np.array(self.pdf_inside)
        values_outside_roi = np.array(self.pdf_outside)
        
        eps_min = np.min(values_inside_roi)
        eps_max = np.max(values_outside_roi)
        
        if eps_min > eps_max:
            return 0
        
        
        relevant_positives = values_inside_roi[np.logical_and(values_inside_roi >= eps_min, values_inside_roi <= eps_max)]
        relevant_negatives = values_outside_roi[np.logical_and(values_outside_roi >= eps_min, values_outside_roi <= eps_max)]
        
        relevant_data = np.concatenate((relevant_positives, relevant_negatives))
             
        overlap_list = np.zeros(relevant_data.size)
        
        for idx, curr_eps in enumerate(relevant_data):
            
            n_false_negatives = np.where(relevant_positives < curr_eps)[0].size
            n_false_positives = np.where(relevant_negatives >= curr_eps)[0].size

            overlap_list[idx] = n_false_negatives + n_false_positives
        
        return np.min(overlap_list)
        


    # optimize the epsilon
    def determin_optimal_epsilon(self):
        """ this function yields the optimal epsilon which minimize the error probability.
        Epsilon will be picked from the values available in the data to avoid unnecesarry calculations.
        
        Returns :
        ---------
            None, however following attributes are newly set :
                
            self.epsilon : float
                values chosen form the self.data, which minimizes the p_err
            self.p_err : float
                error probability, i.e.e the sum of the probability of the missed detections and the false detections
            self.p_miss : float
                the probability of teh missed detections
            self.p_false :
                the probability of the false detections
                
        """
        
        values_inside_roi = np.array(self.pdf_inside)
        values_outside_roi = np.array(self.pdf_outside)
        
        eps_min = np.min(values_inside_roi)
        eps_max = np.max(values_outside_roi)
        
        relevant_positives = values_inside_roi[np.logical_and(values_inside_roi >= eps_min, values_inside_roi <= eps_max)]
        relevant_negatives = values_outside_roi[np.logical_and(values_outside_roi >= eps_min, values_outside_roi <= eps_max)]
        
        relevant_data = np.concatenate(relevant_positives, relevant_negatives)
             
        overlap_list = np.zeros(relevant_data.size)
        
        for idx, curr_eps in enumerate(relevant_data.size):
            overlap_list[idx] = len(np.where(relevant_positives <= curr_eps)) + \
                                len(np.where(relevant_negatives > curr_eps))
        
        return np.min(overlap_list)
        
        
        # set the range of epsilon = min(self.pdf_inside) ... 1 form the self.data
        data_values_all = list(np.array(self.data).flatten())
        data_values_all.sort()
        epsilon_range = [x for x in data_values_all if x >= min(self.pdf_inside)]
        # base to collect the number of pixels for each epsilon value
        pix_missed_list = []
        pix_false_list = []
        p_err_arr = np.zeros((len(epsilon_range)))
        p_miss_list = []
        p_false_list = []
        
        # count pixels 
        for idx, curr_eps in enumerate(epsilon_range):
            # inside the ROI : probabiity of the undetected pixels (below curr_eps) 
            errpix_inside = len([x for x in self.pdf_inside if x <= curr_eps])
            pix_missed_list.append(errpix_inside)
            curr_p_miss = errpix_inside / len(self.pdf_inside)
            p_miss_list.append(curr_p_miss)
            # outside of the ROI : probabiity of the falsely detected pixels (over curr_eps) 
            errpix_outside = len([x for x in self.pdf_outside if x >= curr_eps])
            pix_false_list.append(errpix_outside)
            curr_p_false = errpix_outside / len(self.pdf_outside)
            p_false_list.append(curr_p_false)
            
            # claculate the error probability for the curr_eps
            p_err_arr[idx] = curr_p_miss* self.p_inside + curr_p_false* self.p_outside
            
        # find the optimal epsilon, i.e. where the p_err is minimal
        min_idx = np.where(p_err_arr == p_err_arr.min())[0][0]
        self.epsilon = epsilon_range[min_idx]
        # get the right number of errored pixels inseide/outside of the ROI for the self.epsilon
        self.pix_missed = pix_missed_list[min_idx]
        self.pix_false = pix_false_list[min_idx]
        # get the right probabilities for the self.epsilon
        self.p_err = p_err_arr[min_idx]
        self.p_miss = p_miss_list[min_idx]
        self.p_false = p_false_list[min_idx]
        # self.eps_range = epsilon_range
        


    def get_gcnr(self):
        """ calculate CGNR with the given ROI. If ROI is not specified when the ImageAnalyzer class is called, then 
        self.set_roi() should be called beforehand.
        
        Returns :
        ---------
            CGNR : float, 0...1
                corresponds to how many pixels can be resolved
        """
        # check whether ROI is already specified or not
        if self.roi is None:
            raise AttributeError('ImageQualityAnalyzerGCNR : the ROI should be set. Use set_roi()')
            
        self._calculate_pdf()
        self.ovl = self.get_minimum_overlap()
        #self.determin_optimal_epsilon()
        #self.ovl = self.pix_missed + self.pix_false
        #if self.ovl > len(self.pdf_inside):
        #    self.ovl = len(self.pdf_inside)
        self.gcnr = 1 - self.ovl / 1600# len(self.pdf_inside)
        self.calculate_width()
        return self.gcnr


    def calculate_width(self):
        """
        """
        pdf_outside = np.array(self.pdf_outside)
        vari_outside = np.var(pdf_outside, dtype = np.float64)
        sigma_outside = np.std(pdf_outside, dtype = np.float64)
        self.width_outside = vari_outside + sigma_outside



        
