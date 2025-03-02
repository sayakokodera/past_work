############################# Unittest : data synthesizers grid vs gridless ##################################
############# Unittest for DataGeneratorESE : Grid Size #################
import unittest
import random
import numpy as np
import json

import data_synthesizers_ese
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan

import tools.txt_file_writer as fwriter
import tools.txt_file_reader as freader
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg

import matplotlib.pyplot as plot

plot.close('all')

class TestDataGeneratorGridvsGridless(unittest.TestCase):

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

    ###################################################################################################################
    ###################################################################################################################        
    ### variable settings ###
    def input_variables(self):
        grid = 10**-3* np.array(random.sample(range(1, 10), 3))* ureg.meter
        var_list = ['dxdata']
        var_dict = {
                'dxdata' : grid,
                }
        return var_list, var_dict
   

    def get_3D_positions_with_variable_adjustment(self, test_name, adjusted_param_dict):
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
        
        ### for test (1) : random scan on grid
        ### scan positions ###
        
        ### defect positions ###
        pos_defect_list = []
        for def_idx in range(param_dict['Ndefect']):
            # setup defect positions for gridless data generator
            pos_defect_list.append([x_def[def_idx]* dxdata, y_def[def_idx]* dydata, z_def[def_idx]* dzdata])
            # setup defect_map for fwm grid data generator
            defect_map[int(round(z_def[def_idx])), int(round(x_def[def_idx])), int(round(y_def[def_idx]))] = 1
            
        pos_defect = np.array(pos_defect_list)* ureg.meter
        
        return x_scan, y_scan, pos_scan, pos_defect, defect_map
        
    

    ########### test : scan positions on grid ###############
    def test_ese_grid(self) :
        
        # json data
        fname_const = 'parameter_set/manual_scan_params_constant.json'
        fname_vars = 'parameter_set/manual_scan_params_variables.json'
        # fname_errs = ''
        
        # load json, ds stands for data set
        param_const = json.loads(open(fname_const).read())
        param_vars = json.loads(open(fname_vars).read())
        # param_errs = json.loads(open(fname_errs).read())
        param_errs = {
                'grid_size' : {'values' : [1, 3], 'unit' : ureg.millimeter},
                'Npoint' : {'values' : [50, 100, 150]},
                'roi_distance' : {'values' : [10, 9, 8], 'unit' : ureg.millimeter}
                }
                
        
        dataese_class = data_synthesizers_ese.ErroredDataGrid(var_set = {'dimension' : '1'})
        dataese_class.input_parameter_dataset(param_const, param_vars, param_errs) 
        
        iter_idx = 0
        for curr_var_sub in dataese_class.var_sub:
            for var_sub_idx in range(len(dataese_class.input_param_vars['specimen_variables'][curr_var_sub])):
                dataese_class.var_set.update({
                                curr_var_sub : str(var_sub_idx)
                                })
                dataese_class.register_specimen_variables(dataese_class.input_param_vars)
                
                # (5) iterate over var_main
                for curr_var_main in dataese_class.var_main['values']:
                    # register var_main into the DataGeneratorESE class                                       
                    dataese_class.register_error_source_variables(curr_var_main)
                    # (6) set file names ---> coming soon
                    # (7) call fwm class 
                    dataese_class._call_fwm_class()
                    dataese_class.set_defect_positions()
                    # (8) set fwm parameters
                    dataese_class.input_parameters_into_fwm_class()
                    # (9) configure logger ---> coming soon
                    # (10) get data
                    data = dataese_class.fwm_class.calculate_bscan()
                    # save data
                    fname_data = 'test_errsev_sa_data_' + str(iter_idx) + '.txt'
                    fwriter.write_txt_file(data, 'Ntdata, Nscan', fname_data)
                    iter_idx = iter_idx + 1
                    
                
                with self.subTest():
                    self.assertTrue(np.all(grid_data[:, x_transducer_idx, y_transducer_idx] == 
                                           ms_data))
 

        
if __name__ == '__main__':
    unittest.main()                


