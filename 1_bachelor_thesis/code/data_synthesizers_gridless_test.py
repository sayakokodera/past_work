############################# Unittest : data synthesizers grid vs gridless ##################################

import unittest
import random
import numpy as np
from ultrasonic_imaging_python.forward_models import data_synthesizers_gridless 
import ultrasonic_imaging_python.forward_models.data_synthesizers as data_synthesizers_grid

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
        

        ### pulse models ###
        #pulse_model_name = 'Dirac'

        param_dict = {
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
               #'pulse_model_name' : pulse_model_name,
               'x_def' : x_def,
               'y_def' : y_def,
               'z_def' : z_def,
               'x_scan_gridless' : x_scan_gridless,
               'y_scan_gridless' : y_scan_gridless
               }
        return param_dict

    ###################################################################################################################
    ###################################################################################################################        
    ### variable settings ###
    def input_variables(self):
        var_list = ['dxdata']#, 'dydata']
        var_dict = {
                'dxdata' : np.array([0.0005])*  ureg.meter,#10**-4* np.array(random.sample(range(1, 100), 3))* ureg.meter, #[m] np.array([0.5*10**-3])#
                'dydata' : 10**-4* np.array(random.sample(range(1, 100), 3))* ureg.meter #[m] np.array([0.5*10**-3])#
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
# =============================================================================
#         x_def = 0.01* param_dict['x_def']
#         x_def[0] = param_dict['x_scan_gridless'][0] + 2
#         y_def = 0.01* param_dict['y_def']
#         y_def[0] = param_dict['y_scan_gridless'][0]
#         z_def = 0.01* param_dict['z_def']
#         z_def[0] = np.array([15])
# =============================================================================
        x_def = param_dict['x_def']
        y_def = param_dict['y_def']
        z_def = param_dict['z_def']
        
        ### for test (1) : random scan on grid
        if test_name == 'random_scans_on_grid':
            ### scan positions ###
            x_scan_gridless = dxdata* param_dict['x_scan_gridless']* ureg.meter
            y_scan_gridless = dydata* param_dict['y_scan_gridless']* ureg.meter
            ### defect positions ###
            pos_defect_list = []
            for def_idx in range(param_dict['Ndefect']):
                # setup defect positions for gridless data generator
                pos_defect_list.append([x_def[def_idx]* dxdata, y_def[def_idx]* dydata, z_def[def_idx]* dzdata])
                # setup defect_map for fwm grid data generator
                defect_map[int(round(z_def[def_idx])), int(round(x_def[def_idx])), int(round(y_def[def_idx]))] = 1
                
            pos_defect = np.array(pos_defect_list)* ureg.meter
                        
                
        # for test (2) & (3)
        else :
            ### scan positions ###
            x_scan_gridless = 0.01* dxdata* param_dict['x_scan_gridless']* ureg.meter
            y_scan_gridless = 0.01* dydata* param_dict['y_scan_gridless']* ureg.meter
            ### defect_positions ###
            x_def = dxdata* param_dict['x_def']
            y_def = dydata* param_dict['y_def']
            z_def = dzdata* param_dict['z_def']
            
            pos_defect = np.array(([[x_def, y_def, z_def]]))* ureg.meter
            
            if test_name == 'gridless_scan_defect_on_grid':
                pos_defect_list = []
                for def_idx in range(param_dict['Ndefect']):
                    pos_defect_list.append([dxdata* param_dict['x_def'][def_idx],
                                            dydata* param_dict['y_def'][def_idx],
                                            dzdata* param_dict['z_def'][def_idx]])   
                pos_defect = np.array(pos_defect_list)* ureg.meter                               
        
        return x_scan_gridless, y_scan_gridless, pos_defect, defect_map


    def get_2D_positions_with_variable_adjustment(self, which_test, adjusted_param_dict):
        param_dict = dict(adjusted_param_dict)
        
        ### for test (1) : random scan on grid
        if which_test == 'random_scans_on_grid':
            dxdata = param_dict['dxdata'].to_base_units().magnitude
            dzdata = param_dict['dzdata'].to_base_units().magnitude
            ### scan positions ###
            x_scan_gridless = dxdata* param_dict['x_scan_gridless']* ureg.meter
            ### defect positions ###
            pos_defect_list = []

            for def_idx in range(param_dict['Ndefect']):
                pos_defect_list.append([0.01* dxdata* param_dict['x_def'][def_idx],
                                        0.01* dzdata* param_dict['z_def'][def_idx]])   
                pos_defect = np.array(pos_defect_list)* ureg.meter
                
        # for test (2) & (3)
        else :
            x_scan_gridless = 0.01* param_dict['dxdata']* param_dict['x_scan_gridless']
              
        return x_scan_gridless, pos_defect


    ###################################################################################################################
    ###################################################################################################################    
    ############## 3D ################

    def calc_raw_data_3D(self, adjusted_param_dict, defect_map):
        
        ########## data with grid ##########
        param_dict = dict(adjusted_param_dict)
        griddata = data_synthesizers_grid.DataGeneratorForwardModel3D('3DSingleMedium', 'Dirac')
        griddata.set_forward_model_parameters(param_dict)
        griddata.set_defect_map(defect_map)
        griddata.set_pulse_parameters({'amplitude' : 1})
        grid_raw_data = griddata.get_data()
         
        #grid_param_dict = griddata.get_parameter_dictionary() 
        #print(grid_param_dict)    
        ########## gridless data ##########
        gridlessdata = data_synthesizers_gridless.GridlessDataGenerator3D()
        gridlessdata.register_parameters(param_dict)
        gridless_param_dict = gridlessdata.get_parameter_dictionary()
        print(gridless_param_dict)   
        gridless_raw_data = gridlessdata.calculate_raw_ascans()
         
        return grid_raw_data, gridless_raw_data
    

    ###################################################################################################################
    ############ find 2 neighbors ###########
    def find_2_neighbors(self, gridless_scan_position, rounded_scan_position):
        # returns 2 neighbors of the selected gridless scan position
        # ---> neighbor1 < gridless_scan_position < neighbor2
        upper_neighbor = rounded_scan_position + 1
        lower_neighbor = rounded_scan_position - 1
        
        # case : upper_neighbor is the right neighbor
        neighbor1 = int(rounded_scan_position) # smaller neighbor
        neighbor2 = int(upper_neighbor) # bigger neighbor
        
        # case : lower_neighbor is the right neighbor 
        if abs(upper_neighbor - gridless_scan_position) > abs(lower_neighbor - gridless_scan_position):
            neighbor1 = int(lower_neighbor)
            neighbor2 = int(rounded_scan_position)
        
        neighbors = [neighbor1, neighbor2]
        
        return neighbors
    
    ############ find non-zero elements in the neighbor vectors ###########
    def find_non_zero_elements(self, grid_raw_data, x_neighbors, y_neighbors):
        z_nonzero =[]
        neighbor_not_found = False
        for y_idx in y_neighbors:
            z_nz, x_nz = np.nonzero(grid_raw_data[:, :, y_idx])
            # check whether neighbors are in x_nz        
            neighbors_found_in_x_nz = [x for x in x_neighbors if x in x_nz]
            
            # case : one of the x_neighbors does not exist in x_nz
            if len(neighbors_found_in_x_nz) != len(x_neighbors):
                neighbor_not_found = True
            
            # list up nonzero elements of the neighbors in z_nonzero
            for element in neighbors_found_in_x_nz:
                z_nonzero.append(z_nz[int(list(x_nz).index(element))])
    
        z_smallest = min(z_nonzero)                
        z_largest = max(z_nonzero)
        
        # case : one of the x_neighbors does not exist in x_nz
        if neighbor_not_found == True:
            z_largest = grid_raw_data.shape[0] + 1
            
        return z_smallest, z_largest
            
        
    ###################################################################################################################
    ###################################################################################################################    

    ########### test : scan positions on grid ###############
    def test_random_scan_on_grid(self) :
         ### setup ###
         test_name = 'random_scans_on_grid'
         var_list, var_dict = self.input_variables()
         param_dict = self.create_param_dict(test_name)
         
         
         for var_key in var_list:
             param_dict_copy = dict(param_dict)
             
             for var_value in var_dict[var_key]:
                 param_dict_copy[var_key] = var_value
                 ### variable adjustment : pos_defect, pos_scan ###
                 x_transducer, y_transducer, pos_defect, defect_map = \
                             self.get_3D_positions_with_variable_adjustment(test_name, param_dict_copy)
                 # NEW!! 12.09.18 adjustment to the DataGeneratorGridless regarding 'z_transponder'
                 z_transducer = np.zeros(len(x_transducer))
                 param_dict_copy.update({
                         'pos_defect' : pos_defect,
                         'x_transducer' : x_transducer,
                         'y_transducer' : y_transducer,
                         'z_transducer' : z_transducer
                         })
                 ### get raw data ###
                 grid_raw_data, gridless_raw_data = self.calc_raw_data_3D(param_dict_copy, defect_map)
                 plot.imshow(gridless_raw_data)
                 
                 ### position settings for grid data ###
                 # convert positions into index
                 x_transducer_idx = np.around(x_transducer / param_dict_copy['dxdata']).astype(int)
                 y_transducer_idx = np.around(y_transducer / param_dict_copy['dydata']).astype(int)
 
                 ### find causes ###
                 pos_defect_idx_x = np.around(pos_defect[:, 0] / param_dict_copy['dxdata'])
                 pos_defect_idx_y = np.around(pos_defect[:, 1] / param_dict_copy['dydata'])
                 pos_defect_idx_z = np.around(pos_defect[:, 2] / param_dict_copy['dzdata'])
 
                 with self.subTest(cause_t0 = param_dict_copy['t0'],
                                   defect_position_x = pos_defect_idx_x,
                                   defect_position_y = pos_defect_idx_y,
                                   defect_position_z = pos_defect_idx_z,
                                   scan_pos_x_idx = x_transducer_idx, 
                                   scan_pos_y_idx = y_transducer_idx,
                                   gridless_scan_x = param_dict_copy['x_transducer'],
                                   gridless_scan_y = param_dict_copy['y_transducer']):
                     self.assertTrue(np.all(grid_raw_data[:, x_transducer_idx, y_transducer_idx] == 
                                            gridless_raw_data))
 

    ########### test : grid vs gridless with 1 defect on grid ###############
# =============================================================================
#     def test_grid_vs_gridless_scan_defect_on_grid(self):
#         ### setup ###
#         test_name = 'gridless_scan_defect_on_grid'
#         var_list, var_dict = self.input_variables()
#         param_dict = self.create_param_dict(test_name)
#         
#         
#         for var_key in var_list:
#             param_dict_copy = dict(param_dict)
#             
#             for var_value in var_dict[var_key]:
#                 param_dict_copy[var_key] = var_value
#                 ### variable adjustment : pos_defect, pos_scan ###
#                 x_scan_gridless, y_scan_gridless, pos_defect, defect_map = \
#                             self.get_3D_positions_with_variable_adjustment(test_name, param_dict_copy)
#                 param_dict_copy.update({
#                         'pos_defect' : pos_defect,
#                         'x_transducer' : x_scan_gridless,
#                         'y_transducer' : y_scan_gridless,
#                         'z_transducer' : np.zeros(len(x_scan_gridless))
#                         })
#                 
#                 ### get raw data ###
#                 grid_raw_data, gridless_raw_data = self.calc_raw_data_3D(param_dict_copy, defect_map)
#  
#                 
#                 ### position settings for grid data ###
#                 # convert positions into index
#                 x_scan_rounded_idx = np.around(x_scan_gridless / param_dict_copy['dxdata']).astype(int)
#                 y_scan_rounded_idx = np.around(y_scan_gridless / param_dict_copy['dydata']).astype(int)
#                 print(x_scan_rounded_idx)
#                 print(y_scan_rounded_idx)
#                 ### find depth of the non-zero elements in grid_data
#                 # location of the nonzero elements in zhr gridless raw data
#                 z_nz_gridless, scan_nz_gridless = np.nonzero(gridless_raw_data)
#                 print(z_nz_gridless)
#                 print(scan_nz_gridless)
# 
#                 plot.figure(1)
#                 plot.imshow(gridless_raw_data)
#                 plot.figure(2)
#                 plot.imshow(grid_raw_data[:, x_scan_rounded_idx, y_scan_rounded_idx ])
# 
#                 for idx, scan_idx in enumerate(scan_nz_gridless):
#                     x_neighbors = self.find_2_neighbors(0.01* param_dict_copy['x_scan_gridless'][scan_idx],
#                                                         x_scan_rounded_idx[scan_idx])
#                     y_neighbors = self.find_2_neighbors(0.01* param_dict_copy['y_scan_gridless'][scan_idx], 
#                                                         y_scan_rounded_idx[scan_idx])
#                     
#                     z_smallest, z_largest = self.find_non_zero_elements(grid_raw_data, x_neighbors, y_neighbors)
#                     
#                     with self.subTest(cause_x = scan_idx, cause_z = z_nz_gridless[idx], neighbor1 = z_smallest, 
#                                       neighbor2 = z_largest):
#                         self.assertTrue(z_smallest <= z_nz_gridless[idx] <= z_largest)
# 
# =============================================================================
       
        
if __name__ == '__main__':
    unittest.main()                


