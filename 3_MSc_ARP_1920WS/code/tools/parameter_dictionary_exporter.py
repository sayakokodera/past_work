# -*- coding: utf-8 -*-
import json
import numpy as np

from ..definitions import units
ureg = units.ureg

"""
#===========================#
Parameter Dictionary Exporter
#===========================#

What this classdoes:
    - Check whether each parmeter is with unit or not
    - Separate unit and magnitude from each parameter
    - Check whther the obtained magnitude is array or not (as json does not support np.ndarray)
    - Add unit and magnitude of each parameter to the output dictionary
    - Save the output dictionary as json file

Example usage:
    (1) exporter = ParameterDictionaryExporter(specimen_dict)
    (2) dict_to_export = exporter.convert_dictionary()
    (3) set file name:
        now = datetime.datetime.now() # datetime shoud be imported!
        curr_time = '{}{}{}_{}h{}m{}s'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        fname = 'specimen_params_{}'.format(curr_time)
    (4) exporter.export_dictionary(file_name, False)

"""

class ParameterDictionaryExporter():
    
    def __init__(self, input_dict):
        r""" call super class constructor
        
        Parameters
        ----------
            input_dict : dict (with unit)
                dictionary of parameters with unit
        """        
        super().__init__() 
        self.input_dict = dict(input_dict) # with unit
        self.output_dict = {} # without unit
        

    def separate_unit(self, curr_param):
       r""" Check whether each parametre is with unit or not and separate unit and magnitude from each parameter
       
       Parameters
       ----------
           curr_param : str, 
               the parameter name for checking the unit
               
       Returns
       -------
           magnitude : int, float
           unit : str
               unit for each parameter as string, 'None' if curr_param is unitless
       """
       param = self.input_dict[curr_param]
       if isinstance(param, int) == True or isinstance(param, float) == True:
           magnitude = self.input_dict[curr_param]
           unit = 'None'
       else:
           magnitude = param.magnitude
           unit = str(param.units)
           
       return magnitude, unit
    

    def check_array(self, curr_magnitude):
        r""" Check whether the obtained magnitude is array or not (as json does not support np.ndarray)
        Parameters
        ----------
            curr_magnitude : int, float, np.ndarray (else?)
                the obtained magnitude for the current parameter
                
        Returns
        -------
            output : int, float, list
                (a) if curr_magnitude == int or float -> remains same (i.e. int or float)
                (b) if curr_magnitude == np.ndarray -> converted to a list
        """
        if isinstance(curr_magnitude, np.ndarray) == True:
            output = curr_magnitude.tolist()
        else:
            output = curr_magnitude
        return output
    
    
    def convert_dictionary(self):
        r""" Add unit and magnitude of each parameter to self.output_dict
        
        Returns
        -------
            self.output_dict : 
        """
        for curr_param in self.input_dict:
            magnitude, unit = self.separate_unit(curr_param)
            magnitude = self.check_array(magnitude)
            self.output_dict.update({
                    curr_param : {
                            'magnitude' : magnitude,
                            'unit' : unit
                            }
                    })
        return self.output_dict
        
    
    def export_dictionary(self, fname, export_another_dict, output_dict = None):
        r""" Save the parameter dictionary (without unit, nut containint the information on unit and magnitude) as 
        json file
        
        Parameters
        ----------
            fname : str
            export_another_dict : boolean
                False, when self.output_dict is exported
                True, when another dictionary should be exported
                (e.g. dictionary should contain specimen AND pulse parameters as one dictionary)
            output_dict : dit (without unit)
                default is None, it is only given when export_another_dict == True
        """
        if export_another_dict == True:
            dict_to_export = dict(output_dict)
        else:
            dict_to_export = dict(self.output_dict)
        
        # writing json file
        with open(fname, 'w') as f:
            json.dump(dict_to_export, f)
            
            
            
        
        
        
        
        
        
        
           
           