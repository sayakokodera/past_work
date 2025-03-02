import numpy as np
import abc

class DefectMapSingleDefect(abc.ABC):
    """ Class to generate a defect map for FWM. 
    When a defect is located not exactly on a measurement grid point, rather between grid points,
    we usually round the defect position to the nearest grid point to obtaine an "binary" defect map",
    where the quantization error is inevitable. 
    This class aims to minimize such quantization error by representing the defect map with the
    "energy" of the position.
    The energy is inversely proportional to the distance b/w the defect and a neighboring grid point.
    
    (***) Up to now, this class considers only a single defect case. When there are multiple defects in
    a test object, iterate over the defect position and the resulting defect map is the sum of all defect
    maps.
    
    TODO: write a unittest
    
    """
    
    def __init__(self):
        self.B = None # defect map in matrix form (2D)
        self.c = None # defect map in vector form (1D)
        self.energy = None
        self.p_def = None
        self.neighbors = None # deighboring pixels [[x0, z0], [x1, z1, [x2, z2], [x3, z3]]]
        self.Nt_offset = None # temporal oddset, index (= unitless)
    
    def calculate_distance(self, p):
        return np.sqrt((self.p_def[0] - p[0])**2 + (self.p_def[1] - p[1])**2)
    
    @abc.abstractmethod
    def generate_defect_map_multidim(self):
        """ Generating a defect map as a matrix (for 2D case) for a tensor (3D case)
        """
    
    @abc.abstractmethod
    def convert_defect_map_1D(self):
        """ Converting the multi-dimensional defect map into a vector
        """
    
    def get_defect_map_1D(self):
        self.convert_defect_map_1D()
        return self.b
    
    def get_defect_map_2D(self):
        return self.B
    
    def test_def_map(self):
        nz = np.nonzero(self.b)[0]
        print(self.b[nz] == self.energy)
 

class DefectMapSingleDefect2D(DefectMapSingleDefect):
    
    def __init__(self, p_def, Nx, Nz, dx, dz):
        self.p_def = np.array(p_def) # = [x_def, z_def] [m]#
        self.Nx = int(Nx)
        self.Nz = int(Nz)
        self.dx = float(dx)
        self.dz = float(dz)
        
    def find_neighbors(self):
        # Index for the neighbors
        p_floored = np.floor(self.p_def/np.array([self.dx, self.dz])).astype(int)
        n = p_floored[0]
        m = p_floored[1]
        self.neighbors_idx = np.array([[n, m], [n, m+1], [n+1, m], [n+1, m+1]])
        # Correct position of the neighbors
        self.neighbors = self.neighbors_idx.astype(float)
        self.neighbors[:, 0] = self.neighbors[:, 0]* self.dx
        self.neighbors[:, 1] = self.neighbors[:, 1]* self.dz
        
    def calculate_energy(self): 
        """ Allocate the "energy" of the defect according to the distance b/w the defect and each neighboring point 
        """
        distance = np.zeros(self.neighbors.shape[0])
        for idx in range(4):
            distance[idx] = self.calculate_distance(self.neighbors[idx])
        # Normalize teh distance w/ the total distance
        if 0 in distance:
            idx = list(distance).index(0)
            self.energy = np.zeros(4)
            self.energy[idx] = 1
        else:
            energy_scaled = 1/distance
            # Normalize the energy
            self.energy = energy_scaled/np.sum(energy_scaled)
        
    def generate_defect_map_multidim(self, Nt_offset = 0, col_eliminate = None):
        """ Generating a defect map as a matrix (for 2D case)
        Test
        ----
        nz = np.nonzero(self.def_map)[0]
        print(self.def_map[nz] == self.energy)
        
        Test regarding Nt_offset can be also implemented
        """
        # Adjust to the ROI: in temporal direction
        self.Nt_offset = np.copy(Nt_offset)
        self.p_def[1] = self.p_def[1] - self.Nt_offset* self.dz
        # Find neighboring pixels -> energy assignment
        self.find_neighbors()
        self.calculate_energy()
        # Allocate the corresponding energy to the neighboring points
        self.B = np.zeros(((self.Nz - self.Nt_offset), self.Nx))
        for count, p in enumerate(self.neighbors_idx):
            # p = [x_nb, z_nb]
            self.B[p[1], p[0]] = self.energy[count]
        # Adjust to teh ROI (where the zero-columns are eliminated to accelerate the computation)    
        self.B = self.column_elimination(self.B, col_eliminate)
    
        
    def convert_defect_map_1D(self):
        self.b = self.B.flatten('F')


    def column_elimination(self, B, col_eliminate = None):
        if col_eliminate is None:
            return B
        else:
            print('Column elimination!')
            return np.delete(B, col_eliminate, axis = 1)   


class DefectMapMultiDefect2D():
    
    def __init__(self, p_def_all, w, Nx, Nz, dx, dz):
        self.p_def_all = np.copy(p_def_all) # = [x_def, z_def] [m], all defect positions
        self.w = np.copy(w) # Weighting for each point scatterer  
        self.Nx = int(Nx)
        self.Nz = int(Nz)
        self.dx = float(dx)
        self.dz = float(dz) 
        self.Nt_offset = None 
        self.col_eliminate = None
        self.b_multi = None
        self.dms = None # Single scatterer class
        
        
    def single_defect(self, idx):
        print('current idx = {}'.format(idx))
        print(self.p_def_all[idx][0])
        # Defect map class for single scatterer
        self.dms = DefectMapSingleDefect2D(self.p_def_all[idx][0], self.Nx, self.Nz, self.dx, self.dz)
        self.dms.generate_defect_map_multidim(self.Nt_offset, self.col_eliminate)
        # Defect map of a single scatterer: vector form
        b_single = self.w[idx]* self.dms.get_defect_map_1D()
        return   b_single
    
    
    def generate_defect_map_1D(self, Nt_offset = 0, col_eliminate = None):
        # Setting the ROI parameters
        self.Nt_offset = np.copy(Nt_offset)
        self.col_eliminate = col_eliminate
        # 1D defect maps of many single scatterer
        b_all = np.apply_along_axis(self.single_defect, 0, np.arange(self.p_def_all.shape[0])[np.newaxis, :])
        # 1D defect map of multiple defects = sum of all single scatterers
        self.b_multi = np.sum(b_all, axis = 1)
    
    def get_defect_map_1D(self):
        return self.b_multi
        
    def get_defect_map_2D(self):
        B_multi = np.reshape(self.b_multi, self.dms.B.shape, 'F')
        return B_multi
    

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    Nx = 10
    Nz = 20
    dx = 0.5 #[mm]
    dz = 0.25 #[mm]
    p_def1 = np.array([7*dx, 13*dz])
    p_def2 = np.array([4.8*dx, 10.2*dz])

    dm2d = DefectMapSingleDefect2D(p_def1, Nx, Nz, dx, dz)
    dm2d.generate_defect_map_multidim()
    def_map1 = dm2d.get_defect_map_1D()
    dm2d.test_def_map()

    del dm2d
    dm2d = DefectMapSingleDefect2D(p_def2, Nx, Nz, dx, dz)
    dm2d.generate_defect_map_multidim()
    def_map2 = dm2d.get_defect_map_1D()
    # to test the def_map
    def_map3 = np.zeros(Nx* Nz)
    nz_idx = Nz* int(p_def2[0]/dx) + int(p_def2[1]/dz)
    def_map3[nz_idx] = 1
    print('Def map calculated w/ this calss: {}'.format(np.nonzero(def_map2)))
    print('Def map calculated w/o this calss: {}'.format(np.nonzero(def_map3)))
    
    # ROI adjustment: Nt_offset & column elimination
    Nt_offset = 5
    col_eliminate = np.array([0, 6])
    dm2d.generate_defect_map_multidim(Nt_offset, col_eliminate)
    b_roi = dm2d.get_defect_map_1D()
    print('Def map calculated + ROI adjusted w/ this calss: {}'.format(np.nonzero(b_roi)))
    print('ROI: Nt_offset = {}, eliminated columns = {}'.format(Nt_offset, col_eliminate))
    
    # Multi-defects
    p_def_all = np.array([p_def1, p_def2])
    w = np.array([0.7, 1])
    dmm = DefectMapMultiDefect2D(p_def_all, w, Nx, Nz, dx, dz)
    # W/o columne elimination
    dmm.generate_defect_map_1D(Nt_offset)
    B_m1 = dmm.get_defect_map_2D()
    # W/ column elimination
    dmm.generate_defect_map_1D(Nt_offset, col_eliminate)
    B_m2 = dmm.get_defect_map_2D()
    
    plt.figure(1)
    plt.imshow(B_m1)
    plt.title('W/o col elimination')
    
    plt.figure(2)
    plt.imshow(B_m2)
    plt.title('W/ col elimination')
    

