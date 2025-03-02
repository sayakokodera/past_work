# Main code for thermo stitching -> for running multiple data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap # -> for adjustgin illumination of each image
import os
import glob
from natsort import natsorted # to sort the numbers in strings (s.a. path) correctly (0, 1, 2, ... instead of 0, 1, 10, ...)

from stitching import DataStitcher
from thermography_file_reader import ThermoFileReader
from thermography_preprocessing import ThremoDataPreprocessing, ThermoIlluminationCorrection

# Preprocessing for thermography data --> integrated into stitching
def thermo_preprocessing(arr, pos, new_shape, screw_part=None, transition_mask=None, *args, **kwargs):
    """
    Parameters
    ----------
        screw_type: str (default None)
            either 'shaft' or 'thread' -> for transition handling
            if unspecified, the transition files will not be separated
        transition_mask: boolean array of (8, 8) (None by default)
            This needs to be specified, if screw_type != None.
            It can be obtained via ThermoFileReader
    """
    # (0) Zip
    pos_y, pos_x = pos
    # (1) Center the data
    proc = ThremoDataPreprocessing()
    arr_centered = proc.centering(arr)
    # (2) Remove the irrelevant part
    new_height, new_width = new_shape
    y_start = int(arr.shape[0]/2 - new_height/2)
    x_start = int(arr.shape[1]/2 - new_width/2)
    arr_trimmed = arr_centered[y_start:y_start+new_height, x_start:x_start+new_width]
    # (3) FFT phase reco
    reco = proc.reco_fft_phase(arr_trimmed)[..., 1] # seems only fbin=1 is relevant
    # Zero-mean + standardize
    mean = reco.mean()
    reco = reco - mean
    #reco /= np.abs(reco).max()
    # (4) Screw transition handling
    if (transition_mask[pos_y, pos_x] == True):
        shaft, thread = proc.separate_shaft_thred(reco)
        if screw_part == 'shaft':
            reco = shaft
        elif screw_part == 'thread':
            reco = thread
        else: # i.e. screw_part == None
            reco = np.nan_to_num(shaft) + np.nan_to_num(thread)
    # (5) Normalize for illumination adjustment (using SymLogNorm)
    # Adaptive thresholding selection
    corrector = ThermoIlluminationCorrection()
    corrector.set_parameters(
        p_start = 50.0, 
        num_bins = 401, 
        window_length = 40, 
        stepsize = 5
    )
    th = corrector.adaptive_threshold(reco)
    scale = 1.0
    norm = SymLogNorm(linthresh=th, linscale=scale, vmin=np.nanmin(reco), vmax=np.nanmax(reco), clip=False)
    #SymLogNorm(linthresh=th, linscale=scale, vmin=reco.min(), vmax=reco.max(), clip=False)
    reco_normed = norm.__call__(reco).data

    # Open-To-Do: merge the shaft and thread
    # reco = np.nan_to_num(shaft_normed.data) + np.nan_to_num(thred_normed.data)
    return reco_normed

def glob_all_folders(_path):
    all_files = glob.glob(os.path.join(_path, '*'))
    # filter out irrelevant files, e.g. .zip
    return np.array([file for file in all_files if not file.endswith('.zip')])  


if __name__ == "__main__":
    #----- Parameters: known constants 
    dtype = 'uint16'
    im_height, im_width = 256, 320 # actual size of each data 
    frames = 73 # For stitching, we need to specify the numner of frames ---> output shape! 
    # Measurement adjustment param: vertical (y-direction) & horizontal (x-direction) offset
    # currently: the offset was determined "by eyes" 
    y_offset = 25 # in pixels 
    xmin, xmax = 25, -25
    # Data size AFTER centering and trimming each data 
    seg_height = 209#im_height - y_offset
    seg_width = ThremoDataPreprocessing.surface_projection(angle_start=60, angle_end=120)
    seg_shape = (seg_height, seg_width, frames)
    # Fidelity region ---> determined "by eyes"
    fid_height = int(0.7*seg_height)
    fid_width = ThremoDataPreprocessing.surface_projection(angle_start=75, angle_end=105)

    # Info on the directories
    # path example: "/Volumes/Sandisk_SD/Work/IZFP/ReMachine/Thermografie/4_InspectedSamples/B78/measurements"
    # ---> nomenclature: path = prefix + middle + suffix, with middle = mid1 + mid2
    # Same for all files = prefix and suffix
    prefix = '/Volumes/Sandisk_SD/Work/IZFP/ReMachine/Thermografie/'
    suffix = '/measurements'
    # The middle part is variable -> create an array listing all directories: shape = (len(mid1), len(mid2))
    # mid1 = measurement type (i.e. pre- vs post-damage)
    mid1 = np.array(['4_InspectedSamples', '6_SampleSet_SourceData_Nach_Belastung'])
    # mid2 = screw Nos (e.g. B78, B85 etc)
    # mid2 needs to be globbed + filtered out irrelevant folders (e.g. .zip)
    arr_middle = np.array([glob_all_folders(prefix+mid1[0]), glob_all_folders(prefix+mid1[1])])
    print(arr_middle)
    print(arr_middle.shape)
    print(arr_middle.flatten('C'))
    raise ValueError('Stop!')

    # Iterate over the directories
    for this_middle in arr_middle.flatten('C'):
        print(f'Now = {this_middle}')
        # Instantiate the reader
        reader = ThermoFileReader(prefix+this_middle+suffix)
        which_files = reader.get_files(ret_structured=True)
        
        # Instantiate the stithcer class
        stitcher = DataStitcher()
        stitcher.set_parameters(
            seg_shape=seg_shape, 
            numsegs_per_row=which_files.shape[0], 
            numsegs_per_col=which_files.shape[1]
        )
        stitcher.segment_files = which_files
        stitcher.fidelity_region= (fid_height, fid_width)

        # Stitch
        output_arr = stitcher.stitch(
            func_loader=reader.load_data, 
            func_proc=thermo_preprocessing, 
            trim=False,
            normalize=False,
            weighting=True,
            ymin=y_offset,
            xmin=xmin,
            xmax=xmax,
            new_shape=seg_shape[:2],
            transition_mask=reader.transition_mask
        )

        # Plot + save
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(
            output_arr[:, :, 0], 
            vmin=0.0, vmax=1.0, 
            cmap=plt.cm.gray, aspect=0.5,
        )
        # Ticks
        correct_width = 8*ThremoDataPreprocessing.surface_projection(angle_start=67.5, angle_end=135-22.5) #-> correct width for 360deg
        xtick_offset = int((output_arr.shape[1] - correct_width)/2)
        numsegs_per_col = which_files.shape[1]
        xticks = xtick_offset + stitcher.stepsize_width* np.arange(0, numsegs_per_col+1)
        # tick labels as e.g. ['0°', '45°', ...]
        xticks_labels = np.linspace(0, 360, numsegs_per_col+1, dtype=int).astype(str) + np.repeat('°', numsegs_per_col+1) 
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels)
        
        # Axis setting
        ax.set_xlabel('Rotation')
        # y --> empty
        ax.get_yaxis().set_visible(False)
        
        ax.set_title(f'{this_dir}')
        plt.tight_layout()
        plt.savefig(f'plots/{this_dir}_stitched.png', dpi=250)
        plt.close()
        
        

    
    
    
    

    
    

    
    