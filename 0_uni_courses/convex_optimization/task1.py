r"""
Convex optimization HW: task 1
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
#======================================================================================================= Functions ====#
def calculate_power(mu, sigma, alpha):
    r"""
    Parameters:
    -----------
        mu : int, float
            Water-level for power allocation (i.e. Lagrange multiplier coefficient)
        sigma: int, float
            Noise power
        alpha: np.ndarray with the same size as the number of channel
            Channel gain 
    Return
    ------
        p: np.ndarray with the size of alpha.shape
            Allocated power for each channel
    """
    p = 1/mu - sigma/alpha
    # Constraint: p_i >= 0
    for idx, element in enumerate(p):
        if element < 0:
            p[idx] = 0
    return p


# Functions for bisection method
def bisection_method(fx, xmin_init, xmax_init, target):
    r""" Finding the closest solution, xopt, for the function f(x) = 0 by reducing the range between xmin and xmax 
    which corresponds to f(xmin) <= 0 and f(xmax) >= 0
    Parameters
    ----------
        fx: function
        xmin_init: int, float
            Initial guess for xmin, make sure f(xmin) <= 0
        xmax_init: int, float
            Initial guess for xmax, make sure f(xmax) >= 0
        target: float
            Operation breaks, when abs(f(xopt)) <= target
    Return
    ------
        xopt: float
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
        print('Error = {}'.format(error))
        if np.abs(error) > target:
            # Swapping the either extreme with xopt to reduce the range
            if fx(xmin)* fx(xopt) > 0: 
                xmin = xopt
            else: 
                xmax = xopt
            print('Optimized xmin = {}, xmax = {}'.format(xmin, xmax))
        else:
            print('#======= Target reached!=======#')
            break
    return xopt


#======================================================================================================================#
# parameter
sigma = 1 # noise power
alpha = np.array([4, 3, 2, 1]) # Channel gains for four channels
pmax_all = np.array([0.1, 1, 10]) # Total avialble power
target = 10**(-3)

# Calculate the water-level for different total power  
mu_all = np.zeros(pmax_all.shape[0])
p_all = np.zeros((alpha.shape[0], pmax_all.shape[0]))
for idx, pmax in enumerate(pmax_all):
    # Initial setting
    mu_max = 10**(-3)
    mu_min = 100
    # Continuous funcion we want to solve: p1 + p2 + p3 + p4 - pmax = 0
    def fmu(mu):
        p = calculate_power(mu, sigma, alpha)
        return np.sum(p) - pmax
    
    # Find the closest solution mu via bisection method
    mu_opt = bisection_method(fmu, mu_min, mu_max, target)
    mu_all[idx] = mu_opt
    p_all[:, idx] = calculate_power(mu_opt, sigma, alpha)

print(mu_all)

# Plots
plt.figure(1)
plt.stem(alpha, p_all[:, 0]/pmax_all[0], 'r', markerfmt='rs', label = 'pmax = 0.1')
plt.stem(alpha, p_all[:, 1]/pmax_all[1], 'g', markerfmt='gv', label = 'pmax = 1')
plt.stem(alpha, p_all[:, 2]/pmax_all[2], 'b', markerfmt='bo', label = 'pmax = 10')
plt.legend()
plt.title('Power allocation')
plt.xlabel('Channel gain')
plt.ylabel('Power / pmax')
plt.grid()
plt.savefig('tex/plots/task1_pratio.eps', format = 'eps')


plt.figure(2)
#plt.stem(alpha, p_all[:, 0], 'r', markerfmt='rs', label = 'pmax = 0.1')
plt.stem(alpha, p_all[:, 2], 'b', markerfmt='bo', label = 'pmax = 10')
plt.legend()
plt.title('Power allocation')
plt.xlabel('Channel gain')
plt.ylabel('Power')
plt.ylim(ymax = 3.0)
plt.grid()
plt.savefig('tex/plots/task1_pmax10.eps', format = 'eps')

plt.figure(3)
#plt.stem(alpha, p_all[:, 0], 'r', markerfmt='rs', label = 'pmax = 0.1')
plt.stem(alpha, p_all[:, 1], 'g', markerfmt='gv', label = 'pmax = 1')
plt.legend()
plt.title('Power allocation')
plt.xlabel('Channel gain')
plt.ylabel('Power')
plt.ylim(ymax = 0.5)
plt.grid()
plt.savefig('tex/plots/task1_pmax1.eps', format = 'eps')

plt.figure(4)
plt.stem(alpha, p_all[:, 0], 'r', markerfmt='rs', label = 'pmax = 0.1')
#plt.stem(alpha, p_all[:, 1], 'b', markerfmt='bo', label = 'pmax = 10')
plt.legend()
plt.title('Power allocation')
plt.xlabel('Channel gain')
plt.ylabel('Power')
plt.ylim(ymax = 0.1)
plt.grid()
plt.savefig('tex/plots/task1_pmax01.eps', format = 'eps')
