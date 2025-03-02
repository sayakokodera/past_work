# -*- coding: utf-8 -*-
import numpy as np
"""
With this class, pre-defined unit is added to the given array or dictionary.

"""


class UnitAdderArray():
    r"""
    General Usage: 
        Case: adding unit to a scan position array
        >>> (1) Define the input array: pos_scan = np.array(......)
        >>> (2) Define the base unit: dxdata = 0.5* ureg.millimetre
        >>> (3) Call this class: unitadder = UnitAdderArray(input_arr = pos_scan, unit = dxdata)
        >>> (4) Add unit: x_transducer = unitadder.add_unit_to_array()
    """
    
    def __init__(self, input_arr, unit):
        r""" Constructor
        
        Parameters
        ----------
            input_arr : np.ndarray (1D, 2D...)
            unit : ureg (i.e. with unit)
                e.g. dxdata = 0.5mm -> unit = 0.5* ureg.millimetre
        """
        super().__init__()
        # setup
        self.input_arr = input_arr
        self.unit = unit
                
    def add_unit_to_array(self):
        r""" The proper unit is added to position arrays.
        """
        # check if the given arr_positions are array or not
        if  isinstance(self.input_arr, np.ndarray) == True:
            pass
        else:
            raise TypeError('add_unit_to_array : positions should be given as array')
        
        # add unit
        return np.array(self.input_arr)* self.unit



class UnitAdderDictionary():
    r"""
    General Usage: 
        Case: adding unit to a parameter dictionary
        >>> (1) Define the input dictionary, containing
                - 'magnitude'
                - 'unit
        >>> (2) Define the unit dictionary: 
                e.g. 
                unit_dict = {
                    'c0' : ureg.meter / ureg.second,
                    'fS' : ureg.megahertz,
                    'zImage': ureg.meter,
                    ......
                    }
        >>> (3) Set output dictionary
        >>> (4) Call this class: unitadder = UnitAdderDictionary(input_dict, unit_dict, output_dict)
        >>> (5) Add unit: output_dict = unitadder.add_unit_to_dictionary()
    """
    
    def __init__(self, input_dict, unit_dict, output_dict):
        r""" Constructor
        
        Parameters
        ----------
            input_dict : dict (without unit), 
                containing 'magnitude' & 'unit' 
            unit_dict : dict
                indicating the proper unit for each parameter
            output_dict : dict
                parameter dictionary to obtain containing parameters with unit
        """
        super().__init__() 
        # copy the input
        self.input_dict = dict(input_dict)
        self.unit_dict = dict(unit_dict)
        self.output_dict = dict(output_dict)
        
    
    def check_unit(self, key):
        r""" Unit of the given parameter is examined. if it is correct, the parameter value with the proper unit 
        is returned.
        
        Parameters
        ----------
            key : str,
                the key of the input parameter dictionary to check the unit
        Return
        ------
            param_value : ureg
                parameter value its proper unit
        """
        # case: unitless
        if self.input_dict[key]['unit'] == 'None':
            param_value = self.input_dict[key]['magnitude']
        # case: proper unit is given
        elif self.input_dict[key]['unit'] == str(self.unit_dict[key]):
            param_value = self.input_dict[key]['magnitude']* self.unit_dict[key]
        # case: wrong unit is given
        else:
            raise AttributeError("UnitAdderDictionary : " + key + " should be with " + str(self.unit_dict[key]))
        
        return param_value
        
       
    def add_unit_to_dictionary(self):
        r""" self.output_dict is updated with the proper units.
        
        Return
        ------
            self.output_dict : dict containing parameters with unit
        """
        # iterate over parameters        
        for curr_param in self.input_dict:
            # check the unit and add it to the curr_param
            param_value = self.check_unit(curr_param)
            # update the self.output_dict
            self.output_dict.update({
                    curr_param : param_value
                })
    
        return self.output_dict

            
        
    