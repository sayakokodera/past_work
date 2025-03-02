# -*- coding: utf-8 -*-
from ..definitions import units
ureg = units.ureg

from .unit_adder import UnitAdderDictionary

r"""
%============== json parameter file reader =============%
What this class does:
    - Extract the information regarding magnitude and unit
    - Check unit
    - Add unit
    - Calculate dz
    - Add dz to the parameter dictionary
    - Return the newly set parameter dictionary (with proper unit)
    - Update an already set parameter dictionary (with unit) with additional parameters (without unit)
        
Parameters:
    - input_dict = dictionary obtained from the selected json file, which should contain
                    - magnitude
                    - unit (if no unit is required, this should be declared als "None")
                    -> if the given dictionary does not have either of them, they should be defined beforehand
    - unit_dict = dictionary containing the proper units for each prarameter
    
Example usage:
    Case: get specimen constants from a json file
    (1) Load the json file: 
        json_dict = json.loads(open(fname).read())
        e.g.: fname = '/Users/sako5821/Desktop/git/Master/RP_19SS/Code/parameter_set/params_constant.json'
    (2) Define a unit dictionary with proper units for each parameter:
        e.g.:
            unit_dict = {
                'c0' : ureg.meter / ureg.second,
                'fS' : ureg.megahertz,
                'zImage': ureg.meter,
                'xImage': ureg.meter,
                'yImage': ureg.meter,
                'fCarrier' : ureg.megahertz,
                'dxdata' : ureg.millimeter,
                'dydata' : ureg.millimeter,
                'dzdata' : ureg.millimeter,
                't0' : ureg.second,
                }
    (4) Call this class: 
        param_converter = ParameterDictionaryConverter(json_dict['specimen_constant'], unit_dict)
    (5) Add unit:
        param_converter.add_unit_to_parameters()
    (6) Add dz:
        param_converter.add_dz()
    (7) Get dictionary:
        specimen_constants = param_converter.output_dict
    
"""

class ParameterDictionaryConverter():
    
    def __init__(self, input_dict, unit_dict):
        r""" call super class constructor
        """        
        super().__init__() 
        # copy the dictionaries input into the class
        self.input_dict = dict(input_dict)
        self.unit_dict = dict(unit_dict)
        # the base of the output parameter dictionary (unit will be added)
        self.output_dict = {}
        
       
    def add_unit_to_parameters(self, dictionary_update = False, dict_to_update = None):
        r""" self.output_dict is updated with the proper units.
        
        Parameters
        ----------
            dictionary_update : boolean (default = False)
                True, when this function is used with the update_dictionary function
            ditc_to_update : dict (default = None)
                this will be given, when the output dictionary should be 
        """
        # check, whether it is used for the update_dictionary funtion
        if dictionary_update == True:
            self.output_dict = dict(dict_to_update)
        
        unitadder = UnitAdderDictionary(self.input_dict, self.unit_dict, self.output_dict)
        self.output_dict = unitadder.add_unit_to_dictionary()
        
                
    # add parameters to the given dictionary 
    def update_dictionray(self, dict_to_update):
        r""" This function can be used, when a set of parameter (without unit) shoud be added to a parameter dictionary
        which is alreday set with unit.
        
        Parameters
        ----------
            dict_to_update : dict, containing parameters with the proper unit
        """
        # add unit
        self.add_unit_to_parameters(dictionary_update = True, dict_to_update = dict_to_update)        
        return self.output_dict

                
    def calculate_dz(self, c0, fS):
        r""" The value of dz is calculated(with unit).
        
        Parameters
        ----------
            c0 : speed of sound in ureg (i.e. with unit)
            fS : sampling frequency with ureg (i.e. with unit)
        
        Return
        ------
            dz : spatial grid in ureg (i.e. with unit)
                dz = c0 /(2* fS) with unit
                dz = (input_param_dict['c0'].to_base_units() / (2.0 * input_param_dict['fS'].to_base_units()))
        """
        
        # dz calculation with metre (= base_unit of dz)
        dz_base_unit = c0.to_base_units() / (2.0* fS.to_base_units())
        # convert metre to the proper unit
        dz = dz_base_unit.to(self.unit_dict['dzdata'])
        
        return dz
    
    # for specimen constants
    def add_dz(self):
        r"""dz with the proper unit is added to self.output_dict. This function should be called AFTER c0 and fS are 
        added to self.output_dict (i.e. c0 and fS are with unit).    
        """       
        # check if c0 and fS are in self.output_dict
        if 'c0' in self.output_dict and 'fS' in self.output_dict:
            pass
        else:
            raise AttributeError("ParameterDictionaryConverter : dz cannot be calculated as c0 or fS is missing")
            
        # calculate dz
        c0 = self.output_dict['c0']
        fS = self.output_dict['fS']
        dz = self.calculate_dz(c0, fS)
        
        # add dz to self.output_dict
        self.output_dict.update({
                'dzdata' : dz
                })

    # for pulse constants            
    def modify_pulse_constants(self, fS):
        r""" The parameter dictionary of pulse constantrs may contain only required parameters due to the pulse_models 
        class. With this function 't_puilse' amd 'fS' are added and N_pulse and 'pulse_model_name' are removed from the 
        parameter dictionary (= self.output_dict).
        
        Parameters
        ----------
            fS : ureg (i.e. with unit)
                True, when dz should be calculated and added to the parameter dictionary (i.e. self.outpu_dict)
        
        """
        # save the pulse model name
        self.pulse_model_name = self.output_dict['pulse_model_name']
        # update pulse_params
        self.output_dict.update({
                    'tPulse': self.output_dict['N_pulse'] / fS,
                    'fS': fS,
                    })
        # remove N_pulse and 'pulse_model_name'
        del self.output_dict['N_pulse']
        del self.output_dict['pulse_model_name']
        
        
    # for puler constants
    def get_pulse_model_name(self):
        return self.pulse_model_name
        
    
        
        
        
        
        
        
        