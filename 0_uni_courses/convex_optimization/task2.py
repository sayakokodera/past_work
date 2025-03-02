r"""
Convex Optimization HW: task2
"""
import numpy as np

#======================================================================================================= Functions ====#
def bisection_method(fx, xmin_init, xmax_init, target):
    r""" Finding the closest solution, xopt, for the function f(x) = 0 by reducing the range between xmin and xmax 
    which corresponds to f(xmin) <= 0 and f(xmax) >= 0
    Parameters
    ----------
        fx: function
        xmin_init: int, float, np.ndarray
            Initial guess for xmin, make sure f(xmin) <= 0
        xmax_init: int, float, np.ndarray
            Initial guess for xmax, make sure f(xmax) >= 0
        target: float
            Operation breaks, when abs(f(xopt)) <= target
    Return
    ------
        xopt: float, np.ndarray
            Closest solution for f(x) = 0
    """
    xmin = xmin_init
    xmax = xmax_init
    print('Initial xmin = {}, xmax = {}'.format(xmin, xmax))
    print('fx(xmin) <= 0? {}'.format(fx(xmin) <= 0))
    print('fx(xmax) >= 0? {}'.format(fx(xmax) >= 0))
    if fx(xmin)*fx(xmax) > 0:
        raise AttributeError('Error in fx(xmin)*fx(xmax) > 0')
        
    # Initial error
    error = target + 1
    while np.abs(error) > target:
        xopt = (xmin + xmax)/2 # Calculate the middle point of xmax and xmin
        error = fx(xopt)
        if np.abs(error) > target:
            # Swapping the either extreme with xopt to reduce the range
            if fx(xmin)* fx(xopt) > 0: 
                xmin = xopt
            else: 
                xmax = xopt
        else:
            print('#======= Target reached!=======#')
            break
    return xopt


def flatten_npmat(vec_like):
    vec = vec_like.flatten()
    vec = np.array(vec.T)
    return np.reshape(vec, vec.shape[0])

def calcualte_power(H, v_k):
    r""" Calculate the dot product between H.H and v_k and flatten the answer so that it can be stored directly into an 
    array
    """
    p_k = np.dot(H.H, v_k) # size = (K, 1)
    p_k = flatten_npmat(p_k) # size = K
    p_k = np.abs(p_k)**2
    return p_k


# Weight vectors
def weight_vector(h_k, mat):
    mat_inv = np.linalg.inv(mat)
    return np.dot(mat_inv, h_k)

def scaling_matrix(H, k, lam, method, mu_k = None):
    M = H.shape[0] # Number of TX elements
    K = H.shape[1] # Number of RX
    mat = np.matrix((0 + 0j)* np.ones((M, M)))
    mat += lam* np.identity(M)
    for j in range(K):
        if j != k:
            h_j = H[:, j]
            if method == 1:
                mat += np.dot(h_j, h_j.H)
            elif method == 2:
                mat += mu_k[j]* np.dot(h_j, h_j.H)
    return mat

# Zero-forcing weight vectors
def weight_matrix_ZF(H, pmax):
    K = H.shape[1] # Number of RXs
    # The base of the weight vectors
    Vzf = np.matrix((0 + 0j)* np.ones(H.shape))
    # Beamforming vectors = pseudo-inverse of H.H
    Vzf_scaled = np.linalg.pinv(H.H)
    # Normalize each vector
    for k in range(K):
        v_k = Vzf_scaled[:, k]
        Vzf[:, k] = np.sqrt(pmax/K)* v_k/np.linalg.norm(v_k)
    return Vzf

def calculate_SINR(power, sigma):
    sinr = np.zeros(power.shape[0])
    for k in range(power.shape[0]):
        p_all = np.sum(power[:, k])
        p_interference = p_all - power[k, k]
        sinr[k] = power[k, k] / (p_interference + sigma**2) 
    return sinr

       
def sumrate(sinr):
    r = np.log2(sinr + 1)
    return np.sum(r)

#======================================================================================================================#
# Fixed parameters
M = 16 # number of TX elements
K = 4 # number of RX
sigma = 1.0
Nrealization = 500 # Number of realizations for Monte-Carlo simulation
# Variables
eta_all = np.array([1, 0.1, 0.01, 10**-3, 10**-4])
eta_idx = 2 # selector for eta
eta = eta_all[eta_idx] # Power limit for the interference 
snr_all = np.arange(0, 31, 5)
snr = 30 # dB
pmax = (sigma**2)* 10**(snr/10)

# Targets
target_pk = 10**-5
target_pj = 10**-5
# Base to store the sum-rate
R_meth1 = np.zeros(Nrealization) 
R_meth2 = np.zeros(Nrealization)
R_zf = np.zeros(Nrealization)

for n in range(Nrealization):
    # Base to store the obtained results
    lam_method1 = np.zeros(K)
    lam_method2 = np.zeros(K)
    P1 = np.zeros((K, K)) # Power of method 1 results
    P2 = np.zeros((K, K)) # Power of method 2 results
    
    # Channel
    #np.random.seed(5)
    H = np.matrix(1/np.sqrt(2)*(np.random.random((M, K)) + 1j*np.random.random((M, K))))
    
    #====================== Zero-forcing ======================#
    Vzf = weight_matrix_ZF(H, pmax)
    Pzf = np.abs(np.dot(H.H, Vzf))**2
    sinr_zf = calculate_SINR(Pzf, sigma)
    R_zf[n] = sumrate(sinr_zf)
    
    #====================== Method 1 & 2 ======================#
    # Initial setting
    Mu_init = 1*np.ones((K, K))  
    Mu_init = Mu_init - Mu_init.diagonal()*np.identity(K) # valid for both methods
    # lambda, mu: larger value -> smaller norm(v_k) (because of inversion)
    lam_min = 10
    lam_max = 10**(-8)
    mu_kj_min = 10**6 
    mu_kj_max = 10**(-6) 
    Mu_opt = np.copy(Mu_init)

    for k in range(K): 
        print('======================')
        print('RX No. {}'.format(k))
        print('======================')
        satisfied = False
        h_k = H[:, k]
        n_iter = 0 # iteration number
        
        while satisfied == False:
            # Power constraint
            def flambda(lam):
                mat_k = scaling_matrix(H, k, lam, 2, Mu_opt[:, k])
                v_k = weight_vector(h_k, mat_k)
                power_k = np.linalg.norm(v_k)**2
                return power_k - pmax/K
    
            lam = bisection_method(flambda, lam_min, lam_max, target_pk)
            print('***lam_opt: {}'.format(lam))
            lam_method2[k] = lam
            # Obtained lambda is valid for the method 1, when the Mu == Mu_init
            if n_iter == 0:
                lam_method1[k] = lam
                mat_k = scaling_matrix(H, k, lam, 1)
                v_k = weight_vector(h_k, mat_k)
                P1[:, k] = calcualte_power(H, v_k)
            
            # Interference constraint
            for j in range(K):
                if j != k:
                    h_j = H[:, j]
                    mu_k = np.copy(Mu_opt[:, k])
                    mat_k = scaling_matrix(H, k, lam, 2, mu_k)
                    v_k = weight_vector(h_k, mat_k)
                    interference = np.abs(np.dot(h_j.H, v_k))**2
                    print('Interference: {}'.format(interference))
                    
                    if interference > eta:
                        def fmu(mu_kj):
                            mu_k[j] = mu_kj 
                            mat_k = scaling_matrix(H, k, lam, 2, mu_k)
                            v_k = weight_vector(h_k, mat_k)
                            interference = np.abs(np.dot(h_j.H, v_k))**2
                            return interference - eta
                        
                        mu_kj_opt = bisection_method(fmu, mu_kj_min, mu_kj_max, target_pj) 
                        print('***mu_kj_opt : {}'.format(mu_kj_opt))
                        Mu_opt[j, k] = mu_kj_opt 
    
            # Check if the constraints are still fulfilled
            mat_k = scaling_matrix(H, k, lam, 2, Mu_opt[:, k])
            v_k = weight_vector(h_k, mat_k)
            # Signal/interference power
            P2[:, k] = calcualte_power(H, v_k)
            p_int = np.copy(P2[:, k]) # Interference power
            p_int[k] = 0.0 # "Interference" of the desired signal = 0 
            print('Maximal interference power: {}'.format(max(p_int)))
            if np.abs(np.linalg.norm(v_k)**2 - pmax/K) > target_pk or max(p_int) - eta > target_pj:
                satisfied = False
                print('!!! Not yet satisfied !!!')
                n_iter += 1
            else:
                satisfied = True
                print('!!! Satisfied :) !!!')
                
    # Calculate the SINR for the method 1 & 2
    sinr_meth1 = calculate_SINR(P1, sigma)
    R_meth1[n] = sumrate(sinr_meth1)
    sinr_meth2 = calculate_SINR(P2, sigma)
    R_meth2[n] = sumrate(sinr_meth2)
    
np.save('npy_data/Rzf_{}dB.npy'.format(snr), R_zf)
np.save('npy_data/Rmeth1_{}dB_etaidx{}.npy'.format(snr, eta_idx), R_meth1)
np.save('npy_data/Rmeth2_{}dB_etaidx{}.npy'.format(snr, eta_idx), R_meth2)


