############################# Unittest : SAFT for manual scan ##################################

import unittest
import random
import numpy as np
from ultrasonic_imaging_python.forward_models.data_synthesizers import DataGeneratorForwardModel3D
from ultrasonic_imaging_python.reconstruction.saft_grided_algorithms import SaftEngineProgressiveUpdate3D
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan

from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

import matplotlib.pyplot as plt

plt.close('all')

class TestDataGeneratorGridvsGridless(unittest.TestCase):
    
    ############################################################################################## Parameter Setup #####    
    ######## parameter setup ########
    def create_param_dict(self, test_name):
        ### measurement settings ###
        c0 = 5920* ureg.meter / ureg.second#np.random.uniform(1000, 7000)* ureg.meter / ureg.second # [m/s]
        openingangle = 80 #[grad]
        anglex = 0 #[grad]
        angley = 0 #[grad]
        fS = 80* 10**6* ureg.hertz#np.random.uniform(5, 100)
        tS = (1/fS.magnitude)* ureg.second
        t0 = np.random.uniform(0,10)* tS #np.random.uniform(0,80)* 10**-5* ureg.second # [S], 
        ### dimensions ###
        Ntdata = 50#np.random.randint(20, 100) # excluded t0?
        Nxdata = 30#np.random.randint(10, 30)
        Nydata = 20#np.random.randint(10, 30)
        ### for axis ###
        dxdata = 0.0005* ureg.meter#np.random.uniform(0.0001, 0.01)* ureg.meter 
        dydata = 0.0005* ureg.meter#np.random.uniform(0.0001, 0.01)* ureg.meter #[m], 
        dzdata = 0.5* c0* tS #[m], 
        ### for pulse ###
        fCarrier = 5 * ureg.megahertz
        B = 0.5
        pulse_model_name = 'Gaussian'
                

        ### scan positions ###
        Nscan_gridless = np.random.randint(1, int(min(Nxdata, Nydata)))
        if test_name == 'random_scans_on_grid':
            x_scan_gridless = np.array(random.sample(range(0, Nxdata - 1), Nscan_gridless))
            y_scan_gridless = np.array(random.sample(range(0, Nydata- 1), Nscan_gridless))
        else : 
            x_scan_gridless = np.array(random.sample(range(0, 100*(Nxdata-1)), Nscan_gridless))
            x_scan_gridless[0] = round(0.01* x_scan_gridless[0]) + 0.5
            y_scan_gridless = np.array(random.sample(range(0, 100*(Nydata-1)), Nscan_gridless))
        

       ### defect positions ###
        if test_name == 'random_scans_on_grid':
            Ndefect = np.random.randint(0, Nscan_gridless) #number of defects
            if Ndefect == 0:
                raise ValueError("'Ndefect' : There is no defect inside the specimen")
            #x_def = np.array(random.sample(range(0, 100*(Nxdata-1)), Ndefect))
            x_def = np.array(random.sample(range(0, Nxdata -1), Ndefect))
            x_def[0] = x_scan_gridless[0] + 1
            #y_def = np.array(random.sample(range(0, 100*(Nydata-1)), Ndefect))            
            y_def = np.array(random.sample(range(0, Nydata - 1), Ndefect))
            y_def[0] = y_scan_gridless[0]
            z_def = np.array(random.sample(range(0, Nydata - 1), Ndefect))
            #z_def = np.array(random.sample(range(0, 100*(Ntdata-1)), Ndefect))
            z_def[0] = np.array([20])
            
        elif test_name == 'gridless_scan_defect_on_grid' :
            Ndefect = 1
            x_def = np.array(random.sample(range(0, Nxdata - 1), Ndefect))
            x_def[0] = round(0.01* x_scan_gridless[0]) + 1
            y_def = np.array(random.sample(range(0, Nydata - 1), Ndefect))
            y_def[0] = 0.01* y_scan_gridless[0]#round(0.01* y_scan_gridless[0])
            z_def = np.array(random.sample(range(0, Ntdata - 1), Ndefect))
            z_def[0] = np.array([15])#np.array(random.sample(range(0, Ntdata), Ndefect))
        else : 
            pass
        

        specimen_parameters = {
               'c0' : c0, 
               't0' : t0, 
               'openingangle' : openingangle,
               'anglex' : anglex,
               'angley' : angley,
               'fS' : fS,
               'Ntdata' : Ntdata,
               'Nxdata' : Nxdata,
               'Nydata' : Nydata,
               'Nzreco' : Ntdata,
               'Nxreco' : Nxdata,
               'Nyreco' : Nydata,
               'Nscan_gridless' : Nscan_gridless,
               'dxdata' : dxdata,
               'dydata' : dydata,
               'dzdata' : dzdata,
               'Ndefect' : Ndefect,
               'x_def' : x_def,
               'y_def' : y_def,
               'z_def' : z_def,
               'x_scan_gridless' : x_scan_gridless,
               'y_scan_gridless' : y_scan_gridless
               }
        
        pulse_parameters = {
                'tPulse': 20 / specimen_parameters['fS'],
                'fCarrier' : fCarrier,
                'fS' : specimen_parameters['fS'],
                'B' : B
                }
        
        return specimen_parameters, pulse_parameters, pulse_model_name

    ########################################################################################## Variable Adjustment #####     
    ### variable settings ###
    def input_variables(self):
        var_list = ['dxdata']#, 'dydata']
        var_dict = {
                'dxdata' : np.array([0.0005])*  ureg.meter,#10**-4* np.array(random.sample(range(1, 100), 3))* ureg.meter, #[m] np.array([0.5*10**-3])#
                'dydata' : 10**-4* np.array(random.sample(range(1, 100), 3))* ureg.meter #[m] np.array([0.5*10**-3])#
                }
        return var_list, var_dict
   

    def create_defect_map(self, adjusted_param_dict):
        ### terminology ###
        # adjusted_param_dict = param_dict adjusted to the variable change
        param_dict = dict(adjusted_param_dict)
        dxdata = param_dict['dxdata'].to_base_units().magnitude
        dydata = param_dict['dydata'].to_base_units().magnitude
        dzdata = param_dict['dzdata'].to_base_units().magnitude
        defect_map = np.zeros([param_dict['Nzreco'], param_dict['Nxreco'], param_dict['Nyreco']])
        x_def = param_dict['x_def']
        y_def = param_dict['y_def']
        z_def = param_dict['z_def']
        
        ### defect positions ###
        pos_defect_list = []
        for def_idx in range(param_dict['Ndefect']):
            # setup defect positions for gridless data generator
            pos_defect_list.append([x_def[def_idx]* dxdata, y_def[def_idx]* dydata, z_def[def_idx]* dzdata])
            # setup defect_map for fwm grid data generator
            defect_map[int(round(z_def[def_idx])), int(round(x_def[def_idx])), int(round(y_def[def_idx]))] = 1
        
        return defect_map



    ############################################################################################## Data Generation #####   
    def get_data(self, curr_specimen_parameters, pulse_parameters, defect_map, pulse_model_name):
        
        ########## data with grid ##########
        copy_dict = dict(curr_specimen_parameters)
        fwm_grid = DataGeneratorForwardModel3D('3DSingleMedium', pulse_model_name)
        fwm_grid.set_forward_model_parameters(copy_dict)
        fwm_grid.set_pulse_parameters(pulse_parameters)
        fwm_grid.set_defect_map(defect_map)
        data_3D = fwm_grid.get_data()
        
        Ntdata = copy_dict['Ntdata']
        Nxdata = copy_dict['Nxdata']
        Nydata = copy_dict['Nydata']        
        data_2D = np.zeros((Ntdata, Nxdata* Nydata))
        iter_idx = 0
        for y in range(Nydata):
            for x in range(Nxdata):
                data_2D[:, iter_idx] = data_3D[:, x, y]
                iter_idx = iter_idx + 1
         
        return data_3D, data_2D

    ############################################################################################### Reconstruction #####
    # scan positions for progressive SAFT
    def get_scan_positions(self, reco_param):
        copy_dict = dict(reco_param)
        Nxdata = copy_dict['Nxdata']
        Nydata = copy_dict['Nydata']
        pos_scan = []
        for y in range(Nydata):
            for x in range(Nxdata):
                pos_scan.append([x, y])
        pos_scan = np.array(pos_scan)
        x_scan, y_scan = list(pos_scan[:, 0]), list(pos_scan[:, 1])
        
        return x_scan, y_scan
    
    
    def get_reco(self, data_3D, data_2D, reco_param, x_scan, y_scan):
        copy_dict = dict(reco_param)
        # SAFT for manual scan
        saft_ms = SAFTonGridforManualScan('3DSingleMedium', enable_file_IO = False)
        saft_ms.set_forward_model_parameters(copy_dict)
        reco_ms = saft_ms.get_reco(data_2D, x_scan, y_scan, save_data = False)
        cimg_ms = saft_ms.generate_cscan_format(reco = reco_ms, summation = True, save_data = False)
        plt.figure(1)
        plt.imshow(cimg_ms)
        
        # progressive SAFT
        saft_progressive = SaftEngineProgressiveUpdate3D('3DSingleMedium', enable_file_IO = True)
        saft_progressive.set_forward_model_parameters(copy_dict)
        # base of reco
        reco_prog = np.zeros((copy_dict['Nzreco'], copy_dict['Nxreco'], copy_dict['Nyreco']), dtype = np.float32)
        for curr_x, curr_y in zip(x_scan, y_scan) :
            a_scan = data_3D[:, curr_x, curr_y]
            reco_prog = reco_prog + saft_progressive.get_reconstruction_ascan(curr_x, curr_y, a_scan)
        cimg_prog = saft_ms.generate_cscan_format(reco = reco_prog, summation = True, save_data = False)
        plt.figure(2)
        plt.imshow(cimg_prog)
            
        return reco_ms, reco_prog

    ######################################################################################################### Test #####
    def test_random_scan_on_grid(self) :
         ### setup ###
         test_name = 'random_scans_on_grid'
         var_list, var_dict = self.input_variables()
         specimen_parameters, pulse_parameters, pulse_model_name = self.create_param_dict(test_name)
         
         
         for var_key in var_list:
             param_dict_copy = dict(specimen_parameters)
             
             for var_value in var_dict[var_key]:
                 param_dict_copy[var_key] = var_value
                 ### variable adjustment : pos_defect ###
                 defect_map = self.create_defect_map(param_dict_copy)                   
        
                 ### get data ###
                 data_3D, data_2D = self.get_data(param_dict_copy, pulse_parameters, defect_map, pulse_model_name)
                 
                 ### get reco ###
                 reco_param = dict(param_dict_copy)
                 # scan position setting
                 x_scan, y_scan = self.get_scan_positions(reco_param)
                  
                 # reco param update
                 reco_param.update({
                         'Nxreco' : reco_param['Nxdata'],
                         'Nyreco' : reco_param['Nydata'],
                         'Nzreco' : reco_param['Ntdata'],
                         })
                 reco_ms, cimg_ms, reco_progressive = self.get_reco(data_3D, data_2D, reco_param, x_scan, y_scan)
                 
    
                 with self.subTest(Nxdata = param_dict_copy['Nxdata'],
                                   Nydata = param_dict_copy['Nydata'],
                                   pos_defect = np.nonzero(defect_map)
                                   ):
                     self.assertTrue(np.all(reco_ms == reco_progressive))
 

        
if __name__ == '__main__':
    unittest.main()                


