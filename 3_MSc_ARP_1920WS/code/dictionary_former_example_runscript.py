# -*- coding: utf-8 -*-
"""
Example runscript for dictionary former
"""
import numpy as np
import matplotlib.pyplot as plt

from defect_map_handling import DefectMapSingleDefect2D
from tof_calculator import ToFforDictionary2D
from dictionary_former import DictionaryFormer

#=============================================================================================== Parameter Setting ====#
Nx = 20 # limited due to the opening angle
Nz = 780
Nt = Nz
c0 = 6300 #[m/S]
fS = 80*10**6 #[Hz] 
fC = 5*10**6 #[Hz] 
alpha = 20*10**12 #[Hz]**2
dx = 0.5*10**-3 #[m]
dz = 0.5* c0/(fS)
fwm_param = {
    'Nx' : Nx,
    'Nz' : Nz,
    'Nt' : Nt,
    'c0' : c0, 
    'fS' : fS, 
    'fC' : fC, 
    'alpha' : alpha,
    'dx' : dx
        }
wavelength = 1.26* 10**-3 # [m]
# Time offset
Nt_offset = 480
#t_offset = Nt_offset/fS #[s]

# defect position: p_defect
p_def_idx = np.array([10, 571])
# Convert into defect map
p_def = np.array([p_def_idx[0]*dx, p_def_idx[1]*dz])
dmh = DefectMapSingleDefect2D(p_def, Nx, Nz, dx, dz)
dmh.generate_defect_map(Nt_offset)
defmap_true1 = dmh.get_defect_map()
dmh.generate_defect_map()
defmap_true2 = dmh.get_defect_map()

# Scan positions
p_scan = np.zeros((Nx, 2))
p_scan[:, 0] = np.arange(Nx)*dx
#=================================================================================================== Data modeling ====#
# With t_offset
tofcalc = ToFforDictionary2D(c0, Nx, Nz, dx, dz, p_scan, Nt_offset)
tofcalc.calculate_tof(calc_grad = False)
tof1 = tofcalc.get_tof()
# Dictionary
dformer = DictionaryFormer(Nz, fS, fC, alpha, Nt_offset)
dformer.generate_dictionary(tof1)
H1 = dformer.get_SAFT_matrix()
a1 = np.dot(H1, defmap_true1)
A1 = np.reshape(a1, ((Nt - Nt_offset), Nx), 'F')
del H1, a1

# Without t_offset
tofcalc = ToFforDictionary2D(c0, Nx, Nz, dx, dz, p_scan)
tofcalc.calculate_tof(calc_grad = False)
tof2 = tofcalc.get_tof()
# Dictionary
dformer = DictionaryFormer(Nz, fS, fC, alpha)
dformer.generate_dictionary(tof2)
H2 = dformer.get_SAFT_matrix()
a2 = np.dot(H2, defmap_true2)
A2 = np.reshape(a2, (Nt, Nx), 'F')
del H2, a2

# Evaluation
print(np.array_equal(A1, A2[Nt_offset:, :]))

plt.figure(1)
plt.imshow(A1)
plt.title('With offset')

plt.figure(2)
plt.imshow(A2[Nt_offset:, :])
plt.title('Without offset')
