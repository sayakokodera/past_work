import numpy as np
import scipy.optimize as sco

def quasi_Newton(fx, grad_fx, x0, epsilon, Niteration):
    """ Solve a minimization problem via quasi Newton method (BFGS algorithm).
    
    Parameters
    ----------
    fx : callable function
        Objective function
    grad_fx : callable function
        Gradient of the objective function
    x0 : ndarray
        Initial guess
    epsilon : float
        Target value (for break condition)
    Niteration : int
        Break condition 
        
    Return
    ------
    x_star : ndarray
        Solution for the objective function
        
    Notes
    -----
    This is based on the BFGS algorithm of the quasi Newton method.
    See Wright and Nocedal, 'Numerical Optimization', 1999, pg. 140-141.
    """
    # Call sco.BFGS
    bfgs = sco.BFGS()
    bfgs.initialize(x0.shape[0], 'hess')
    
    # Values based on the initial guess x0
    gradf = grad_fx(x0)
    H = np.identity(x0.shape[0])
    x = x0
    
    for curr_iter in range(Niteration):
        #print('Iteration No. {}'.format(curr_iter))
        p = -np.dot(H, gradf)
        alpha = sco.line_search(fx, grad_fx, x, p)[0]
        if alpha is None:
            alpha = np.dot(gradf.T, p)
        print(alpha)
        x_star = x + alpha* p
        print(fx(x_star))
        # The break condition is not satisfied -> update
        if fx(x_star) > epsilon:   
            gradf_star = grad_fx(x_star)
            bfgs.update(x_star - x, gradf_star - gradf)
            H = bfgs.get_matrix()
            x = x_star
            gradf = gradf_star
        # The break condition is satisfied
        else:
            break
    return x_star


# Example usage
if __name__ == '__main__':
    
    def fx(x):
        return 3*x[0]**2 + 2*x[0]*x[1] + 1.5*x[1]**2
    
    def grad_fx(x):
        return np.array([6*x[0] + 2*x[1], 2*x[0] + 3*x[1]])
    
    x0 = np.array([1, 1])
    epsilon = 10**-6
    Niteration = 15
    print(fx(x0))
    x_opt = quasi_Newton(fx, grad_fx, x0, epsilon, Niteration)
    print(fx(x_opt))
