# -*- coding: utf-8 -*-
################## data synthesizer for error source evaluation (ESE) ##################
import numpy as np
import abc
import logging
# data generator for gridded scans
from ultrasonic_imaging_python.forward_models.data_synthesizers_gridless import GridlessDataGenerator3D
# data generator for manual scans
from ultrasonic_imaging_python.forward_models.data_synthesizers_manual_scan import DataGeneratorManualScanWithoutError
from ultrasonic_imaging_python.forward_models.data_synthesizers_manual_scan import DataGeneratorManualScanPositionError
from ultrasonic_imaging_python.forward_models.data_synthesizers_manual_scan import DataGeneratorManualScanZscanError
# reconstruction 
from ultrasonic_imaging_python.reconstruction.saft_for_manual_scan import SAFTonGridforManualScan
# tools
from ultrasonic_imaging_python.definitions import units
ureg = units.ureg


class DataGeneratorESE(abc.ABC):
    
    # constructor
    def __init__(self):
        """ Constructor
        
        Parameters
        ----------
            None

        """
        super().__init__() 
        
        # input parameters
        self.input_param_const = {} # constant parameters
        self.input_param_vars = {} # variables, self.var_sub should be selected from here
        self.input_param_errs = {} # error sources 
        
        ### parameter setting ###
        self.specimen_parameters = {}
        self.pulse_parameters = {}

        # dictionary for examining the unit of the input dictionary/file
        self.unit_dict = {
                'c0' : ureg.meter / ureg.second,
                'fS' : ureg.megahertz,
                'zImage': ureg.meter,
                'xImage': ureg.meter,
                'yImage': ureg.meter,
                'fCarrier' : ureg.megahertz,
                'dxdata' : ureg.millimeter,
                'dydata' : ureg.millimeter,
                't0' : ureg.second,
                'grid_size' : ureg.millimeter,
                'base_grid_size' : ureg.millimeter,
                'grid_reco' : ureg.millimeter,
                'zscan_err' : ureg.millimeter,
                'pos_defect' : ureg.millimeter,
                't_offset' : ureg.second,
                }
        # keys included in the json file of specimen_variables 
        self.variables = ['dimension', 'defect_map', 'scan_path', 'base_grid_size', 'scan_area_size', 'Npoint',
                          'grid_reco', 'sigma', 'pos_defect', 'openingangle']

        
        ### variables : defined in each inherited class ###
        # values of error sources (list) for iteration
        self.var_main = []
        # variables (dict) for changing scenarios/conditions of the evaluation
        self.var_sub = {}
        # variable dict, for register_specimen_variables, containing key and its variable set (e.g. '0')
        self.var_set = {}
        
        # data synthesizers : defined in each inherited class
        self.fwm_class = None
        
    def _set_file_names(self):
        pass

########################################################################################## logger configuration #######     
    def _configure_logger(self, fname_log, fname_json_const, fname_json_var, var_set):
        """ configuration for logging the program/data. The data is logged, only when the data is saved. 
        (i.e. save_data == True)

        Parameters
        ----------
            fname_log : str, name of the log file
            fname_json_const : str, name of the json file containing constant parameters
            fname_json_var: str, name of the json file, containing variables
            var_set : dict, suggesting which variable sets were selected for generating the data
        """
        
        # setup info to pack into the logger
        self.fname_json_const = str(fname_json_const)
        self.fname_json_var = str(fname_json_var)
        # add var_set here!!!!!
        
        # logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
        
        self.file_handler = logging.FileHandler(fname_log)
        self.file_handler.setFormatter(formatter)
        
        self.logger.addHandler(self.file_handler)

        
################################################################################## parameter setting #######
    def input_parameter_dataset(self, parameter_constant, parameter_variables, error_source_dataset):
        """ 
        What this function does
        -----------------------
            - set input parameters into the selected class.
            - set self.var_main (list)
            
        Parameter
        ---------
            parameter_constant : dict
                containing the unit and the value of constant parameters (such as c0) 
            parameter_variables : dict
                containing variable sets for vaying the measurement scenario
                each variable set should provide the info on its unit and its value
                keys = ['dimension', 'scan_path', 'grid_size', 'roi_distance' ....]
            error_source_dataset : dict
                containing the unit and the values of error sources  
        """
        # input parameters
        self.input_param_const = parameter_constant
        self.input_param_vars = parameter_variables
        self.input_param_errs = error_source_dataset
        
        # set var_main
        self.set_var_main()
        
        
    @abc.abstractclassmethod
    def set_var_main(self):
        """ since var_main varies among classes, self.var_main should be set individually in each class
        """

    ### setup for specimen constant ###
    def register_constant_parameters(self, parameter_constant):
        """ with this funciton the parameter dictionary can be obtained from the selected json file and
            added to the self.specimen_parameters or the self.pulse_parameters (both with unit)
        
        Parameters 
        ----------
            specimen_constant : dict containing following two dictionaries...
                specimen_constant : dict
                    containing values of c0, fS, openingangle, anglex, angley, t0, xImage, yImage, zImage
                pulse_params_default : dict
                    containing info on pulse_model_name, number of pulse (N_pulse) and fCarrier
            
        Reutuns
        -------
            None
                
        newly set parameters (not returned) 
        -----------------------------------
            self.specimen_parameters : dict (with unit)
                containing c0, fS, openingangle, anglex, angley, t0, xImage, yImage, zImage
            self.raw_pulse_params : dict (with unit)
                containing tPulse, fCarrier, fS and B
            pulse_model_name : str
                
        """
        # ureg is not converted properly from json files, thus unit should be added here for FWM
        self._update_dictionary_with_unit(parameter_constant['specimen_constant'], self.specimen_parameters)
        self.specimen_parameters  = self.get_specimen_parameters()
        self._update_dictionary_with_unit(parameter_constant['pulse_params_default'], self.pulse_parameters)
        self.pulse_parameters =self.get_pulse_parameters()
    
        
        # update pulse_params
        self.pulse_parameters.update({
                    'tPulse': self.pulse_parameters['N_pulse'] / self.specimen_parameters['fS'],
                    'fS': self.specimen_parameters['fS'],
                    })
        # pulse_parameters may contain only required parameters due to the pulse_models class, 
        # thus N_pulse and 'pulse_model_name' should be eliminated
        del self.pulse_parameters['N_pulse']
        del self.pulse_parameters['pulse_model_name']
        
        #register pulse model name
        self.pulse_model_name = parameter_constant['pulse_params_default']['pulse_model_name']
        
        
    ### setup for specimen variables ###    
    def register_specimen_variables(self, specimen_variables):
        """ with this funciton the specien variables are set according to the selected variable-set and
            added to the self.specimen_parameters.           
            For properly using this function, self.var_set_dict should be set beforehand.

            self.var_set_dict = dict,
                suggesting which variable-set should be used for selected variables.
                This dictionary should contain both default variable values (wchih is preset in __init__)
                and var_sub (which is set after the for-loop over var_sub).
        
        Parameters 
        ----------
            specimen_variables : dict
                containing dimension (Nxdata, Nydata & Ntdata), defect_map and scan_path_files (t0 = optional),
                etc...
                which should be specified before input (i.e. updated with eaach iteration).
                e.g. : specimen_variables[dimension] should contain only one set of Nxdata, Nydata and Ntdata
            
        Reutuns
        -------
            None
                
        newly set parameters (not returned) 
        -----------------------------------
            self.specimen_parameters : dict (with unit)
                containing c0, fS, openingangle, anglex, angley, t0, xImage, yImage, zImage + 
                Nxdata, Nydata, Ntdata, defect_map(unitless) and scan_path_files
            
        """
        for item in self.variables:
            if item in self.var_set:
                curr_var_set = self.var_set[item] # indicates which var_set is currently chosen
                if item == 'dimension':
                    self.specimen_parameters.update({
                        'Nxdata': int(specimen_variables['specimen_variables']['dimension'][curr_var_set]['Nxdata']), 
                        'Nydata': int(specimen_variables['specimen_variables']['dimension'][curr_var_set]['Nydata']),
                        'Ntdata': int(specimen_variables['specimen_variables']['dimension'][curr_var_set]['Ntdata'])
                        })
                elif 'base_unit' in specimen_variables['specimen_variables'][item] :
                    value = specimen_variables['specimen_variables'][item][curr_var_set]
                    unit = specimen_variables['specimen_variables'][item]['base_unit']
                    if unit == str(self.unit_dict[item]) :
                        if item == 'base_grid_size' :
                            self.specimen_parameters.update({
                                'dxdata': value* self.unit_dict[item], 
                                'dydata': value* self.unit_dict[item],
                                })
                        else :
                            self.specimen_parameters.update({
                                item : value* self.unit_dict[item]
                                })
                    else :
                        raise AttributeError('DataGeneratorESE : base grid size should be with millimeter')
                
                else :
                    self.specimen_parameters.update({
                        item : specimen_variables['specimen_variables'][item][curr_var_set]
                        })
        
    ### setup for error sources ###
    @abc.abstractmethod
    def register_error_source_variables(self, curr_var_main):
        """
        what this function should do:
            1. 
        """
        

    ###### check unit of parameters in json data #########
    def _update_dictionary_with_unit(self, from_which_dict, to_which_dict):
        """ 

        Parameters
        ----------
            from_which_dict : dict,
                the parameter dictionary (created from json files) from which the parameters are taken
            to_which_dict : dict
                the parameter dictionray to which these parameters with unit are added
        
        Returns
        -------
            to_which_dict(with unit)
        
        """
        copy_dict = from_which_dict
        
        
        for key in copy_dict:
            param_data = from_which_dict[key]
            if isinstance(param_data, dict):
                item_with_unit = self._check_units(key, param_data['unit'], param_data['magnitude'])
                to_which_dict.update({
                        key : item_with_unit
                        })
            else:
                to_which_dict.update({
                        key : param_data
                    })


    def _check_units(self, key, input_unit, input_value):
        """ with this function the format (i.e. unit) of the parameters in json files is examined and for the 
        given item the unit is added to the input value.
        
        Parameters
        ----------
            key : key of the copy_dict suggesting which parameter should be added to the parameter dictionary
            input_unit : unit given in json files
            input_value : value to be added to the parameter dictionary
            
        Returns
        -------
            input value with the input_unit
        
        """
        if input_unit == str(self.unit_dict[key]):
            return input_value* self.unit_dict[key]
        else:
            raise AttributeError("ErrorSourceEvaluator : " + key + " should be with " + str(self.unit_dict[key]))
    
    
    
    @abc.abstractmethod            
    def call_fwm_class(self):
        """
        """
    
    @abc.abstractclassmethod
    def input_parameters_into_fwm_class(self):
        """
        """
        
    @abc.abstractclassmethod    
    def _define_scan_positions(self):
        """
        """

    def get_specimen_parameters(self):
        # or in the run script call ErrorSourceEvaluator.specimen_parameters
        return self.specimen_parameters
    
    def get_pulse_parameters(self):
        return self.pulse_parameters


##################################################################################################### Grid Size ####### 
class ErroredDataGrid(DataGeneratorESE):
    """ this is a class for evaluating, how coarse the measurement grid can be.
    
    Scenario
    --------
        setup :
            base_grid_size = 0.5mm (or a wave length) = finest grid in our setting
            (-> should be input as specimen_const?)
            ---> dxdata, dydata = base_grid_size, const
            ---> instead, x_transucer & y_transducer are varied reflecting the 'grid_size' (= var_main)
            Nxdata, Nydata = size of the data, based on the base_grid_size
                           = ROI[mm] = Nxdata* dxdata x Nxdata* dydata
            
            
        
    Variables
    ---------
        var_main : 
            'grid_size' with ureg.millimeter
            ---> scan positions (x_transducer, y_transducer) are varied!
                dxdata, dydata = base_grid_size = remains same!!! 
        var_sub : 
            Ndefects : number of scatterers in the specimen
        var_set : includes
            'dimension' = '0'(default)
            'base_grid_seize' = '0' = 0.5mm (default)
            
            
    """
    # constructor
    def __init__(self, var_set = None):
        super().__init__()

        # setting up the var-subs
        self.var_sub = ['openingangle']
        if var_set == None :
            self.var_set = {
                    'dimension' : '4',
                    'base_grid_size' : '0',
                    'defect_map' : '0'
                    }
        else :
            self.var_set = var_set
        

    def set_var_main(self):
        """ with this function var_main is set for the input_parameter_dataset function. 
            var_main in this class is grid size which corresponds to the scan positions
            (i.e. x_transducer, y_transducer).
        """
        # setting up the var_main
        self.var_main = self.input_param_errs['grid_size'] # = dict, containing 'values' & 'unit'
            
    
    def register_error_source_variables(self, curr_value):
        """
        what this function should do:
            1. adjust the scan positions according to the current grid_size
            2. register scan positions in self.specimen_parameters
        """
        # check unit ----> should be modified!!!!
        grid_with_unit = self._check_units('grid_size', self.var_main['unit'], curr_value)
        self.specimen_parameters.update({
                'grid_size' : grid_with_unit
                })
                
        sampling_grid = int(curr_value / self.input_param_vars['specimen_variables']['dimension']['base_grid'])        
        x_scan_range = np.arange(0, self.specimen_parameters['Nxdata'], sampling_grid)
        y_scan_range = np.arange(0, self.specimen_parameters['Nydata'], sampling_grid)        
        self.x_scan_idx = []
        self.y_scan_idx = []
        
        for y in y_scan_range :
            for x in x_scan_range :
                self.x_scan_idx.append(x)
                self.y_scan_idx.append(y)
        
    def _define_scan_positions(self):  
        """ with this function the specimen_parmaeters dictionary is updated with the scan positions generated with
        the registered grid in self.register_error_source_variables
        """
        self.specimen_parameters.update({
                'x_transducer' : np.array(self.x_scan_idx)* self.specimen_parameters['dxdata'],
                'y_transducer' : np.array(self.y_scan_idx)* self.specimen_parameters['dydata'],
                'z_transducer' : np.zeros(len(self.x_scan_idx))* ureg.meter
                })
        

    def call_fwm_class(self):
        self.fwm_class = DataGeneratorManualScanWithoutError()


    def input_parameters_into_fwm_class(self):
        """
        What this function does :
            - set specimen parameters into DataGeneratorForwardModel3D
            - set pulse parameters into DataGeneratorForwardModel3D
            - create a defect map for fwm
            - set the defect map into DataGeneratorForwardModel3D
        """
        self._define_scan_positions()
        self.fwm_class.register_parameters(self.specimen_parameters)
        self.fwm_class.set_pulse_parameters(self.pulse_model_name, self.pulse_parameters)        

    def get_cimg(self, data, summation):
        r""" this function provides a cscan format of data using the function generate_cscan_format in 
        DataGeneratorGridless. The newly converted data of 3D format can be also now obtained with 
        fwm_class.data_3D.
        """
        cimg = self.fwm_class.generate_cscan_format(data = data, summation = summation, 
                                                    Nxdata = self.specimen_parameters['Nxdata'], 
                                                    Nydata = self.specimen_parameters['Nydata'], 
                                                    x_scan_idx = self.x_scan_idx, y_scan_idx = self.y_scan_idx,
                                                    save_data = False)
        return cimg

    
################################################################################################# Scan Positions #######
class ErroredDataScanPositions(DataGeneratorESE):
    """ this is a class for evaluating, how much impact the wrong information on the scan positions has on the 
    reconstruction quality.
    
    Scenario
    --------
        setup :
            comming soon
            
            
        
    Variables
    ---------
        var_main : 
            'sigma' : float, unitless
            ---> sigma of the positional error, varies the standard deviation for ErrorGenerator
            
        var_sub : 
            'Npoint' : number of points picked up from the san path
            'grid_reco' : determines the level of quantization (---> currently cannot be varied)
            
        var_set : includes
            'dimension' = '0'(default)
            'base_grid_seize' = '0' = 0.5mm (default)
            'Npoint' = '0' = 400 (default)
            'grid_reco' : '0' = 0.5mm (default),
            'sigma' : '0' = 1 (default)
            
            
    """
    # constructor
    def __init__(self, var_set = None):
        super().__init__()
        # setting up the var-subs
        self.var_sub = ['Npoint']#, 'grid_reco']
        if var_set == None :
            self.var_set = {
                    'dimension' : '4',
                    'base_grid_size' : '0',
                    'defect_map' : '0',
                    'Npoint' : '0',
                    'grid_reco' : '0',
                    'sigma' : '0'
                    }
        else :
            self.var_set = var_set
            
        # scan positions
        self.x_transducer_list = []
        self.y_transducer_list = []
        
        # error configuration
        self.err_range = None
        self.seed_value = None
        self.sigma = None
    
    def set_var_main(self):
        """ with this function var_main is set for the input_parameter_dataset function. 
            var_main in this class is sigma.
            Error will be generated with np.random.normal(mean, sigma)
        
        """
        self.var_main = self.input_param_errs['sigma']

    def register_error_source_variables(self, curr_value):
        """
        what this function should do:
            1. add unit to err_range for error configuration
            2. the other necessary values for error configuration (such as sigma, seed_value, mean) are also set to the 
            self.specimen_parameters
        """
        wavelength = (self.specimen_parameters['c0']/self.pulse_parameters['fCarrier']).to_base_units()
        self.err_range = self.specimen_parameters['dxdata'] #wavelength
        print('the err_range : {}mm'.format((self.err_range).magnitude))
        self.seed_value = self.input_param_const['seed']['default_value']
        self.sigma = curr_value  
        print('the current sigma : {}'.format(self.sigma))
        


    def call_fwm_class(self):
        self.fwm_class = DataGeneratorManualScanPositionError()
        

    def _define_scan_positions(self, seed_value):
        """ with this function random point are chosen within the given dimension.
        As a result, self.x_transducer_list and self.y_transducer_list are set and updated in the 
        self.specimen_parameters.
        
        Parameters
        ----------
            seed_value : int
        """       
        self.fwm_class.pick_scan_positions(self.specimen_parameters['Npoint'], seed_value, 
                                           self.specimen_parameters['dxdata'],
                                           self.specimen_parameters['Nxdata'], self.specimen_parameters['Nydata'])


    def input_parameters_into_fwm_class(self):
        """
        what this function should do :
            - set specimen parameters and scan positions in the DataGeneratorManualScan 
            - define random Npoint scan positions
            - set pulse parameters in the DataGeneratorManualScan
        """
        # define scan positions using Npoint and seed value        
        self._define_scan_positions(self.seed_value)
        
        # input specimen parameters into DataGenerator
        self.fwm_class.set_measurement_parameters(params_from_json = False,
                                                  measurement_params = self.specimen_parameters, 
                                                  posscan_from_json = False,
                                                  pos_scan = self.fwm_class.pos_scan)
        # input pulse parameters into DataGenerator
        self.fwm_class.set_pulse_parameters(self.pulse_model_name, self.pulse_parameters)     

        


################################################################# Contact Pressure : propagation time (= z_scan) #######
class ErroredDataPressureZscan(DataGeneratorESE):
    """ this is a class for evaluating, how much impact the contact pressure can have on the imaging quality of the 
    reconstruction.
    
    Scenario
    --------
        setup :
            Data are generated on grid with random positions of z_transducer(z_scan) which represents the change of the 
            propagation time due to the varying contact pressure. The z_scan error can be obtained with the 
            DataGeneratorManualScanZscanError class.
            
            
        
    Variables
    ---------
        var_main : 
             'sigma' : float, unitless
            ---> sigma of the z_scan change, varies the standard deviation for ErrorGenerator
            
        var_sub : 
            'defect_map' : list, indicates where the defects are located
            
        var_set : includes
            'dimension' = '0'(default)
            'base_grid_seize' = '0' = 0.5mm (default)
            'Npoint' = '3' = 1600 (default)
            'grid_reco' : '0' = 0.5mm (default),
            'sigma' : '0' = 1 (default)
            
            
    """
    
    # constructor
    def __init__(self, var_set = None):
        super().__init__()
        
        self.var_sub = ['openingangle']
        if var_set == None :
            self.var_set = {
                    'dimension' : '4',
                    'base_grid_size' : '0',
                    'defect_map' : '0',
                    'grid_reco' : '0',
                    }
        else :
            self.var_set = var_set
            
        # scan positions
        self.x_transducer_list = []
        self.y_transducer_list = []
        
        # error configuration
        self.err_range = None
        self.seed_value = None
        self.sigma = None
        
    
    def set_var_main(self):
        """ with this function var_main is set for the input_parameter_dataset function. 
            var_main in this class is 'zscan_err' (representing the error range of the z_scan)
        
        """
        self.var_main = self.input_param_errs['sigma']
 
    def register_error_source_variables(self, curr_value):
        """
        what this function should do:
            1. define err_range = dz for error configuration
            2. the other necessary values for error configuration (such as sigma, seed_value, mean) are also set to the 
            self.specimen_parameters
            3. setup scan positions
        """
        dz = (self.specimen_parameters['c0'].to_base_units() / 
              (2.0 * self.specimen_parameters['fS'].to_base_units()))
        self.err_range = 1* ureg.millimetre#dz
        self.seed_value = self.input_param_const['seed']['default_value']
        self.sigma = curr_value  


    def _define_scan_positions(self):
        """ with this function scan positions (x, y, z) are set. In this class, they are on grid.
        The error of the z_transducer will be set automatically with self.input_parameters_into_fwm_class with the 
        given error configuration. 
        (to be exact, it'll be set in the DataGeneratorManualScanZscanError calss with set_measurement_parameters)
        As the paramete_parser needs the information on 'z_transducer', the initial value of the z_transducer 
        (i.e. zeros) are set in this function.
        """
            
        x_scan_range = np.arange(0, self.specimen_parameters['Nxdata'])
        y_scan_range = np.arange(0, self.specimen_parameters['Nydata']) 
        
        self.x_transducer_list = []
        self.y_transducer_list = []
        
        for y in y_scan_range :
            for x in x_scan_range :
                self.x_transducer_list.append(x)
                self.y_transducer_list.append(y)
                
        
        self.specimen_parameters.update({
                'x_transducer' : np.array(self.x_transducer_list)* self.specimen_parameters['dxdata'],
                'y_transducer' : np.array(self.y_transducer_list)* self.specimen_parameters['dydata'],
                'z_transducer' : np.zeros(len(self.x_transducer_list))* ureg.meter
                })

    def call_fwm_class(self):
        self.fwm_class = DataGeneratorManualScanZscanError()
        

    def input_parameters_into_fwm_class(self):
        """
        what this function should do :
            - set specimen parameters and scan positions in the DataGeneratorManualScan 
            - define random Npoint scan positions
            - set pulse parameters in the DataGeneratorManualScan
        """
        # define scan positions using Npoint and seed value        
        self._define_scan_positions()
        
        # input specimen parameters into DataGenerator
        self.fwm_class.set_measurement_parameters(params_from_json = False,
                                                  measurement_params = self.specimen_parameters, 
                                                  posscan_from_json = False,
                                                  pos_scan = None)
        # input pulse parameters into DataGenerator
        self.fwm_class.set_pulse_parameters(self.pulse_model_name, self.pulse_parameters)     




############################################################################################################ SA ####### 
class ErroredDataScanArea(DataGeneratorESE):
    """ this is a class for evaluating, how far the scan area (SA) can be from the scatterer.
    
    Scenario
    --------
        setup :
            size of test object = ROI = [Nxdata, Nydata]
            size of scan area = [Nxsa, Nysa] (assumption Nysa == Nydata)
            a single defect is located at the right end of the ROI (near Nxdata)
            
        SA is initially placed on the left side end of the ROI, i.e. only inspects the region of 
        [0:Nxsa-1, Nysa-1], and moved along x-axis to approach the scatterer.
        The max value of 'pos_sa' is Nxdata - Nxsa + 1.
        
    Variables
    ---------
        var_main : 
            'pos_sa' without unit indicating where the scan are starts, iteration range = Nxdata 
            e.g. 'pos_sa' = 1 ---> SA = [1:Nxsa, 0:Nysa-1]
            ---> 'pos_scan' is updated with iteration
        var_sub : 
            'scan_area_size' : representing the Nxsa 
            'grid_size' : grid size (based on 'millimeter', but unitless)
        var_set :
            'dimension' : '0'
            'scan_area_size' : default '0', iterate when it it's var_sub
            'grid_size' : default '0', iterate when it it's var_sub
            
    """
    # constructor
    def __init__(self, var_set = None):
        super().__init__()
        
        # setting up the var-main relevant parameter (pos_scan)
        self.pos_scan = []
        # setting up the var-subs
        self.var_sub = ['scan_area_size', 'grid_size'] # should be loaded from errs_json
        if var_set == None :
            self.var_set = {
                    'defect_map' : '0',
                    'dimension' : '0',
                    'scan_area_size' : '0', # default value
                    'base_grid_size' : '0' # default value
                    }
        else :
            self.var_set = var_set
    
    def set_var_main(self):
        """ with this function var_main is set for the input_parameter_dataset function. 
            var_main in this class is 'pos_sa'
        
        """
        self.var_main = self.input_param_errs['pos_sa'] # = dict, containing 'values' & 'unit' 
    
    def register_error_source_variables(self, curr_value):
        """
        this function sets the self.scan_pos according to the pos_sa (i.e. curr_value)
            3. append self.pos_scan
        """
        self.pos_scan = []
        Nydata = self.specimen_parameters['Nydata']
        Nxsa = self.specimen_parameters['scan_area_size']
        for y_scan in range(Nydata):
            for x_scan in range(curr_value, Nxsa + curr_value):
                self.pos_scan.append([x_scan, y_scan])
        self.pos_scan = np.array(self.pos_scan)* self.specimen_parameters['grid_size']* ureg.millimeter
        self.specimen_parameters.update({
                'pos_scan' : self.pos_scan
                })
        
    def _define_scan_positions(self):
        pass

    def call_fwm_class(self):
        self.fwm_class = GridlessDataGenerator3D()
    


    def input_parameters_into_fwm_class(self):
        """
        what this function should do :
            - set specimen parameters and scan positions in the DataGeneratorManualScan 
            - set pulse parameters in the DataGeneratorManualScan
        """
        # input specimen parameters into DataGenerator
        self.fwm_class.set_measurement_parameters(params_from_json = False,
                                                  measurement_params = self.specimen_parameters, 
                                                  posscan_from_json = False,
                                                  pos_scan = self.pos_scan)
        
        
        # input pulse parameters into DataGenerator
        self.fwm_class.set_pulse_parameters(self.pulse_model_name, self.pulse_parameters)
        

    
######################################################################################################## Npoint #######
class ErroredDataNpoint(DataGeneratorESE):
    # constructor
    def __init__(self):
        super().__init__()
        
    def set_var_main(self):
        """ with this function var_main is set for the input_parameter_dataset function. 
            var_main in this class is .......
        
        """
        pass

    def register_error_source_variables(self, err_sources):
        """
        what this function should do:
            1. 
        """
        pass

    def _define_scan_positions(self):
        pass


    def call_fwm_class(self):
        pass


    def input_parameters_into_fwm_class(self):
        pass

    
################################################################################################## Distribution #######
class ErroredDataDistribution(DataGeneratorESE):
    # constructor
    def __init__(self):
        super().__init__()
    
    def set_var_main(self):
        """ with this function var_main is set for the input_parameter_dataset function. 
            var_main in this class is .......
        
        """
        pass
 
    def register_error_source_variables(self, err_sources):
        """
        what this function should do:
            1. 
        """
        pass


    def _define_scan_positions(self):
        pass


    def call_fwm_class(self):
        pass
        

    def input_parameters_into_fwm_class(self):
        pass
