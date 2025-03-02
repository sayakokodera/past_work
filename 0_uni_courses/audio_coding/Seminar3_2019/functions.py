import numpy as np
import scipy.signal as scs


class MDCTFilter():

    def __init__(self):
        self.H = None
        self.N = None
        self.L = None
        self.M = None


    def analysis_filter(self, input_signal, Nsubband):
        """
        (1) baseband fct (h_n)
        (2) iteration over N
        (3) calc. h_k (w/ the size of L)
        (4) y_k = convolve b/w x (w/ the size M) and h_k -> returns the array w/ the size of (M + L -1)
        (5) Y_k = z-transform of y_k
        (6) DS of Y -> with the size of ((M + L - 1)/N, N)
        """
        x_n = np.array(input_signal) # copy of the input signal
        self.N = int(Nsubband)
        self.L = 2 * self.N # Filter length
        self.M = x_n.shape[0] # the length the input signal 
        Y = np.zeros((self.M, self.N)) # matrix containing all filter output BEFORE DS w/ size: M x N
        self.H = np.zeros((self.L, self.N)) # base of the filter matrix
        # Baseband function h_n
        h_n = np.sin(np.pi / self.L * (np.arange(self.L) + 0.5))
        # Convolution b/w h_k and x_n
        for k in range(self.N):
            # Filter for the k-th 
            h_k = h_n* np.sqrt(2.0/ self.N) * np.cos(np.pi/ self.N * (k + 0.5)* (np.arange(self.L) + 0.5 - self.N/ 2))
            h_k = h_k[::-1] # flip w.r.t the time axis
            self.H[:, k] = h_k
            # k-th filter output
            y_k = np.convolve(h_k, x_n) # size: (M + L -1)
            #y_k = y_k[self.L-1:] # Discard the t<0 part (because of the causality) -> no delay in the reco
            y_k = y_k[:self.M] # remove the fade out part -> size: M x 1
            Y[:, k] = y_k
            
        # Down sampling (DS)
        Y_ds = np.zeros(((int(Y.shape[0]/self.N) + 1), self.N))
        for col in range(self.N):
            Y_ds[:, col] = Y[::self.N, col] # size: int(M/N) x N
            
        return Y_ds

        
    def synthesis_filter(self, Y_ds):
        """
        (1) Upsampling
        (2) Iteration over N
        (3) Convolve with h_k
        (4) Sum up all filter output of each subband  
        """ 
        K = Y_ds.shape[0]* self.N
        Xhat = np.zeros((self.M, self.N))
        for k in range(self.N):
            # Upsampling
            y_k = np.zeros(K)
            y_k[::self.N] = Y_ds[:, k]
            # Convolvtion
            g_k = self.H[::-1, k] # filp the analysis filter fct         
            xhat_k = np.convolve(g_k, y_k)
            xhat_k = xhat_k[:self.M] # remove the fade out part -> size: M x 1
            Xhat[:, k] = xhat_k
        # Sum up all filter output of each subband (i.e. all col. vectors)
        xhat = np.sum(Xhat, axis = 1)
        return xhat, Xhat


    def get_analysis_IR(self):
        return self.H



