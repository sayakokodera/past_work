import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
from natsort import natsorted # to sort the numbers in strings (s.a. path) correctly (0, 1, 2, ... instead of 0, 1, 10, ...)

class ThermoFileReader():
    """
    >>> Data info:
        (a) typical relative path (= path_rel)
            ---> .../Thermografie/4_InspectedSamples/B78/measurements/Sergey_Schrauben_Ansicht1 (5).ITvisPulse
            Notation in this class:
                path_rel_1 = '.../Thermografie/4_InspectedSamples/B78/measurements'
                path_rel_2 = 'Sergey_Schrauben_Ansicht1 (fileNo).ITvisPulse', with the correct fileNo
                path_rel = path_rel_1 + '/' + path_rel_2
        (b) xml file (contains measurement info)
            {path_rel}/measurement.xml
        (c) bin file (thermo data)
            {path_rel}/data.bin
        (d) fileNo and the measurement structure: (y_axis, x_axis) = (8, 8)-images
            for x = 0 deg, y = top --> bottom: [0 ... 7]
            for x = 45 deg, y = top --> bottom: [17 ... 10]
            for x = 90 deg, y = top --> bottom: [18 ... 25]
            for x = 135 deg, y = top --> bottom: [35 ... 28]
            for x = 180 deg, y = top --> bottom: [36 ... 43]
            for x = 225 deg, y = top --> bottom: [53 ... 46]
            for x = 270 deg, y = top --> bottom: [54 ... 61]
            for x = 315 deg, y = top --> bottom: [71 ... 64] 
    """
    
    def __init__(self, _path_rel_1):
        """
        Parameters
        ----------
            _path_rel_1: str
                The relative path to the data
                See the data info (a) above for further detail
        """
        # Instantiate the global object
        self.path_rel_1 = _path_rel_1
        self.arr_fileNos = None
        self.arr_path_rel = None
        # A boolean array indicating which row is transition between the shaft and the thread
        self.transition_mask = self.compute_transition_mask()

    #=========================================
    # Formulate the path to the data
    # -> this includes 
    # (1) identifying the fileNo for each (row, col)-combination => self.structure_files
    # (2) return the path to each data (as a structured array or a list) => self.get_files
    #=========================================   
    def construct_arr_fileNos(self):
        # Structure the file number in the array format ---> this array contains only the file numbers! 
        # Instantiate
        Ny, Nx = 8, 8
        self.arr_fileNos = np.zeros((Ny, Nx), dtype=int)
        counter = 0
        for x in range(Nx):
            # check if x is even or odd
            # (i) even case: ascending in y
            if x%2 == 0:
                start = counter
                this_col = np.arange(start, start+Ny)
            # (ii) odd case: descending in y
            else:
                start = counter+2 # !! start != counter !!! (because two files inbetween are irrelevant)
                this_col = np.flip(np.arange(start, start+Ny))
            # Update arr_numbers with parentheses
            self.arr_fileNos[:, x] = this_col
            # Update the counter
            counter = start + Ny

    def extract_number(self, s):
        # A fnction to extraact the number from each folder name
        # (e.g.) 'Sergey_Schrauben_Ansicht1 (2).ITvisPulse' --> 2
        start = s.find('(')  # Find the position of the first '('
        end = s.find(')', start)  # Find the position of the first ')' after '('
        if start != -1 and end != -1:  # Check if both '(' and ')' exist
            return int(s[start + 1:end])  # Extract and convert the number to int
        else:
            return 0  # Return None if no parentheses are found
    
    def structure_files(self):
        """
        Output
        ------
            arr_paths: np.array, object
                An array containing the paths to the measurements
                This array is formatted for stitching, see the data info (d) above
        """
        # (1) Structure the file number in the array format --> generate self.arr_fileNos
        self.construct_arr_fileNos()
        
        # (2) List all 'Sergey_Schrauben_Ansicht1 XXX.ITvisPulse' folders with glob
        # Get all files in the folder (recursively if needed)
        all_files = glob.glob(os.path.join(self.path_rel_1, '*'))
        # Filter out irrelevant files
        # !!! Weirdly, there are files starting with '.Sergey_Schrauben ...' --> which we sould ignore
        all_files = [file for file in all_files if file.endswith('.ITvisPulse')]  

        # (3) Extract the numbers from ALL the files (including the ones irrelevant for stitching)
        vec_fileNos = np.vectorize(self.extract_number)(all_files)

        # (4) Structure the file names as an array according to self.arr_fileNos
        arr_paths = np.zeros(self.arr_fileNos.shape, dtype=object)
        for y in range(self.arr_fileNos.shape[0]):
            for x in range(self.arr_fileNos.shape[1]):
                # Find the index of all_files, whose file No corresponds to self.arr_fileNos[y, x]
                index = np.where(vec_fileNos == self.arr_fileNos[y, x])[0][0]
                arr_paths[y, x] = all_files[index]
        return arr_paths
        

    def get_files(self, ret_structured):
        if ret_structured == True:
            return self.structure_files()
        else:
            # !!! Weirdly, there are files starting with '.Sergey_Schrauben ...' --> which we sould ignore
            files_all = natsorted(
                [os.path.join(self.path_rel_1, f) for f in os.listdir(self.path_rel_1) if f.startswith('Sergey')]
            )
            # Put 'Sergey_Schrauben_Ansicht1" as the fileNo = 0
            files_all = [files_all[-1]] + files_all[:-1]
            return files_all

    def get_fileNo(self, pos_y, pos_x):
        if self.arr_fileNos is None:
            _ = self.structure_files()
        return self.arr_fileNos[pos_y, pos_x]

    #=========================================
    # Determine where the transition data are
    #=========================================
    def compute_transition_mask(self):
        # A boolean array indicating which row is transition between the shaft and the thread
        mask = np.zeros((8, 8),  dtype='bool')
        # row = 4 --> transition
        mask[4, :] = True
        return mask
        
    #=========================================
    # Extract the time frame info
    #=========================================
    def extract_info_from_xml(self, _path_rel, tag_name):
        fname = f'{_path_rel}/measurement.xml'
        # Parse the XML file
        tree = ET.parse(fname)
        root = tree.getroot()
    
        # Find all elements with the specified tag name (which can be more than one elements)
        elements = root.findall(f".//{tag_name}")
        # Extract only the relevant information (as a list)
        info = [element.text for element in elements]
        return info

    
    def get_frames_from_xml(self, _path_rel):
        # Get the correspondint elements
        info = self.extract_info_from_xml(_path_rel, tag_name='CountFrames')
        # Extract the time frames ---> here, info contains only 1 element, which is the number of frames
        frames = int(info[0])
        return frames

    #=========================================
    # Load
    #=========================================
    def load_data(self, path_rel=None, pos=None, 
                  ymin=None, ymax=-1,
                  xmin=0, xmax=-1, 
                  ret_fileNo=False, frames=None,
                  *args, **kwargs):
        r"""
        This function is written s.t. it can be used as stand alone (i.e. just loading a single image)
        or used as a part of stitching. 
            (a) stand alone:
                you can specify either the position of the image, or provide the 
                whole path (i.e. path_rel) to the data 
            (b) used for stitching
                For stitching purpose, the path (= path_rel) needs to be specified (we chose this way
                s.t. the stitching class remains as generic as possible). 
        
        Parameters
        ----------
            path_rel: str (None by default)
                Path to the data
            pos : array (pos_y, pos_x) (None by default)
                Position of the desired file in the array of (8, 8)
                See the data info (d) above for further detail
        """
        # (0.1) Specify the path to the data
        if path_rel is None:
            if self.arr_path_rel is None:
                self.arr_path_rel = self.get_files(ret_structured=True)
            pos_y, pos_x = pos
            path_rel = self.arr_path_rel[pos_y, pos_x]
        
        # (0.2) Check if the frames need to be determined 
        if frames is None:
            frames = self.get_frames_from_xml(path_rel)

        # (0.3) Extract the measurment parameter = image width, height, dtype
        width = int(self.extract_info_from_xml(path_rel, tag_name='Width')[0])
        height = int(self.extract_info_from_xml(path_rel, tag_name='Height')[0])
        dtype = self.extract_info_from_xml(path_rel, tag_name='BinMode')[0].lower()

        # (0.4) Check if the array params are specified
        if ymin is None:
            ymin = 0
            print(f'ThermoFileReader.load_data: ymin is not specified, so it is set to {ymin}')
            
        # (1) Load
        data = np.fromfile(f'{path_rel}/data.bin', dtype=dtype)
        # Formatting the raw data as a 3D array
        # Desired format: axis 0 = height, axis 1 = width, axis 2 = frames
        # How the data "changes" from fast to slow: width -> height -> frames
        # (1) reshape with 'F' in the order: width -> height -> frames 
        data_tens = np.reshape(data, (width, height, frames), order='F')
        # (2) swap the first two axes  
        data_tens = np.swapaxes(data_tens, 0, 1)
        # Trim the top 10% pixels (where we don't see anything) 
        data_tens = data_tens[ymin:ymax, xmin:xmax, ...]
        if ret_fileNo == False:
            return data_tens
        else:
            fileNo = self.get_fileNo(pos_y, pos_x)
            return data_tens, fileNo


        