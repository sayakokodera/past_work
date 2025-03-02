# -*- coding: utf-8 -*-
import numpy as np
import abc
from .unit_adder import UnitAdderArray
from ..definitions import units
ureg = units.ureg
"""
Defect map converter
"""

class DefectMapConverter(abc.ABC):    
    r""" Abstract base class for defect map conversion.
    
    """
    def __init__(self):
        r""" Constructor
        
        Parameters
        ----------

            None
            
        -------
        """
        super().__init__()
        
    
    def add_unit(self, arr_position, unit):
        r""" The proper unit is added to the given position array using the function add_unit_to_array.
        
        Parameters
        ----------
            unit : ureg (i.e. with unit)
                e.g. dxdata = 0.5mm -> unit = 0.5* ureg.millimetre
            arr_position : np.ndarray (1D, 2D...)
        
        Return
        ------
            pos_with_unit : array with unit
        """
        unitadder = UnitAdderArray(arr_position, unit)
        pos_with_unit = unitadder.add_unit_to_array()
        return pos_with_unit
    
    @abc.abstractclassmethod
    def get_pos_defect(self):
        r""" Convert the given defect_map into pos_scan
        """
    
class DefectMapConverter3D(DefectMapConverter):    
    r""" 3D defect map conversion
    
    """
    def __init__(self, defect_map, dx_data, dy_data, dz_data):
        r""" Constructor
        
        Parameters
        ----------
            defect_map : np.ndarray
                e.g. np.array([[x0, y0, z0], [x1, y1, z1]...])
            dx_data, dy_data, dz_data : ureg (with unit)
                Make sure, that they are with the same unit (e.g. ureg.millimetre)
        -------
        """
        super().__init__()
        self.defect_map = defect_map
        self.dx = dx_data.magnitude
        self.dy = dy_data.magnitude
        self.dz = dz_data.magnitude
        self.unit = dx_data.units
        # set pos_defect
        self.pos_defect = np.zeros(self.defect_map.shape)
        
    def get_pos_defect(self):
        r""" Add teh proper unit to each element of the defect map
        
        Return
        ------
            self.pos_defect : np.array with unit
                e.g. [[x0, y0, z0], [x1, y1, z1], ...]
        """
        # with ureg, adding unit to an entire array is possible, but adding unit to rows/columns is not possible
        # x
        self.pos_defect[:, 0] = self.defect_map[:, 0]* self.dx
        # y
        self.pos_defect[:, 1] = self.defect_map[:, 1]* self.dy
        # z
        self.pos_defect[:, 2] = self.defect_map[:, 2]* self.dz
        # add unit
        self.pos_defect = self.add_unit(self.pos_defect, self.unit)
        
        return self.pos_defect
    

class DefectMapConverter2D(DefectMapConverter):    
    r""" 2D defect map conversion
    
    """
    def __init__(self, defect_map, dx_data, dz_data):
        r""" Constructor
        
        Parameters
        ----------
            defect_map : np.ndarray
                e.g. np.array([[x0, y0, z0], [x1, y1, z1]...])
            dx_data, dy_data, dz_data : ureg (with unit)
                Make sure, that they are with the same unit (e.g. ureg.millimetre)
            
        -------
        """
        super().__init__()
        self.defect_map = defect_map
        self.dx = dx_data.magnitude
        self.dz = dz_data.magnitude
        self.unit = dx_data.units
        # set pos_defect
        self.pos_defect = np.zeros(self.defect_map.shape)
        
    def get_pos_defect(self):
        r""" Add teh proper unit to each element of the defect map
        
        Return
        ------
            self.pos_defect : np.array with unit
                e.g. [[x0, z0], [x1, z1], ...]
        """
        # with ureg, adding unit to an entire array is possible, but adding unit to rows/columns is not possible
        # x
        self.pos_defect[:, 0] = self.defect_map[:, 0]* self.dx
        # z
        self.pos_defect[:, 1] = self.defect_map[:, 1]* self.dz
        # add unit
        self.pos_defect = self.add_unit(self.pos_defect, self.unit)
        
        return self.pos_defect
        
        
class DefectMapGenerator(abc.ABC):
    r""" Convert a pos_defect array into a defect_map 
    """
    def __init__(self):#, pos_defect, dx, dz, Nx, Nz):
        r""" Constructor
        
        Parameters
        ----------
            pos_defect : np.ndarray with unit!
                The information on teh defect positions as 
                [[x_def0, y_def0, z_def0], [x_def1, y_def1, z_def1]]
            dx : ureg (with unit) in [m] or [mm]
                Stepsize for x direction
            dz : ireg (with unit) in [m] or [mm]
                Stepsize for z direction
            Nx : int
                Dimension of the defect map (x)
            Nz : int
                Dimension of the defect map (z)
        """
        super().__init__()
        self.pos_defect = None#pos_defect
        self.dx = None#dx
        self.dz = None#dz
        self.Nx = None#Nx
        self.Nz = None#Nz
        
    
    @abc.abstractclassmethod    
    def get_index(self):
        r""" Quantize the pos_defect wit the proper stepsize
        """
        
        
    @abc.abstractclassmethod
    def get_defect_map(self):
        r""" Convert the pos_defect info (with unit) into a defect_map
        """
    @abc.abstractclassmethod  
    def vectorize_defect_map(self):
        r""" Vectorize the defect_map
        
        Returns
        -------
            defect_vec : np.array with the size of Nz x Ncolumn (2D case = Nx, 3D case = Nx*Ny)
                Vectorized defect map, containing 1 at the defect position and else are 0
                The order of the vectorization is Nz-> Nx -> Ny
        """
        

class DefectMapGenerator2D(DefectMapGenerator):
    def __init__(self, pos_defect, dx, dz, Nx, Nz):
        r""" Constructor
        Parameters
        ----------
            pos_defect : np.ndarray with unit!
                The information on teh defect positions as 
                [[x_def0, y_def0, z_def0], [x_def1, y_def1, z_def1]]
            dx : ureg (with unit) in [m] or [mm]
                Stepsize for x direction
            dz : ireg (with unit) in [m] or [mm]
                Stepsize for z direction
            Nx : int
                Dimension of the defect map (x)
            Nz : int
                Dimension of the defect map (z)
        """
        super().__init__()
        self.pos_defect = pos_defect
        self.dx = dx
        self.dz = dz
        self.Nx = Nx
        self.Nz = Nz
        # defect_map
        self.defect_map = np.zeros((self.Nz, self.Nx))
        
        
    def get_index(self, *args):
        r""" Quantize the pos_defect wit the proper stepsize
        
        Returns
        -------
            x_defidx : np.ndarray with the size of Ndefect x 1
                Containing the index of the defect positions (x)
            z_defidx : np.ndarray with the size of Ndefect x 1
                Containing the index of the defect positions (z)
        """
        x_defidx = np.around(self.pos_defect[:, 0] / self.dx).astype(int) 
        z_defidx = np.around(self.pos_defect[:, 1] / self.dz).astype(int) 
        
        return x_defidx, z_defidx
    
    
    def get_defect_map(self, *args):
        r""" Convert the pos_defect info (with unit) into a defect_map
        
        Return
        ------
            self.defect_map : np.ndarray with the size of Nz x Nx
                Containing either 0 or 1
                Elements in the defect_map is set to 1, where a defect is located
        """
        # Get index
        x_defidx, z_defidx = self.get_index()
        # Get the base of the defect_map
        for curr_x, curr_z in zip(x_defidx, z_defidx):
            self.defect_map[curr_z, curr_x] = 1
        
        return self.defect_map
    

    def vectorize_defect_map(self, *args):
        r""" Vectorize the defect_map
        
        Returns
        -------
            defect_vec : np.array with the size of Nz x Ncolumn (2D case = Nx, 3D case = Nx*Ny)
                Vectorized defect map, containing 1 at the defect position and else are 0
                The order of the vectorization is Nz-> Nx -> Ny
        """
        defect_vec = np.reshape(self.defect_map, (self.Nz* self.Nx, 1), order = 'F')
        
        return defect_vec
        
        