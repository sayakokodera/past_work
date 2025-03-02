import numpy as np
import matplotlib.pyplot as plt


class PulseFormerGabor():
    """ Pulse former for a sinlge defect and single scan position. 
    """
    def __init__(self, Nt, fS, fC, alpha, r = 0):
        r""" Constructor
        Parameters
        ----------
            r: float, -1 < r < 1, unitless (0 by default)
                Chirp rate which determines the asymmetric level of the envelope
                r = (alpha_l - alpha_r) / (alpha_l + alpha_r)
                    alpha_l = decay rate for the left side
                    alpha_r = decay rate for the right side
                (=> r = 0 means that the enevelope is symmetric)
        """
        self.Nt = int(Nt)
        self.fS = float(fS)
        self.fC = float(fC)
        self.alpha = float(alpha)
        self.r = float(r)
        self.env = None
        self.grad_env = None
        self.ip_pulse = None
        self.q_pulse = None
        self.grad_ip = None
        
    def set_t(self, tau):
        self.t = (np.arange(0, self.Nt)/ self.fS - tau)
        self.omega = 2* np.pi* self.fC
        
    def calculate_envelope(self, tau):
        r""" Calculate the envelope of the ultrasonic pulse. 
        Parameters
        ----------
            tau: float [S, no ureg]
                ToF, which corresponds to the peak position of the envelope
        """
        self.set_t(tau)
        # Symmetric envelope
        if self.r == 0:
            self.env = np.exp(- self.alpha* self.t**2)
        # Asymmetric envelope
        else:
            tau_idx = int(tau*self.fS)
            self.env = np.zeros(self.Nt) # base
            # Left side: t < tau
            self.env[:tau_idx] = np.exp(- self.alpha* (1 + self.r)* self.t[:tau_idx]**2)
            # Right side: t >= tau 
            self.env[tau_idx:] = np.exp(- self.alpha* (1 - self.r)* self.t[tau_idx:]**2)
        
    def calculate_grad_envelope(self, grad_tau):
        self.grad_env = 2* self.alpha* self.t* grad_tau* self.env
        
    def calculate_ip_pulse(self, tau, phi = 0):
        self.calculate_envelope(tau)
        self.phi = phi
        self.ip_pulse = self.env* np.cos(self.omega* self.t + np.deg2rad(self.phi))
        
    def calculate_q_pulse(self):
        self.q_pulse = self.env* np.sin(self.omega* self.t + np.deg2rad(self.phi))
        
    def calculate_grad_ip(self, grad_tau):
        self.calculate_q_pulse()
        first_term = 2* self.alpha* self.t* grad_tau* self.ip_pulse
        second_term = self.omega* grad_tau* self.q_pulse
        self.grad_ip = first_term + second_term
        
    def get_envelop(self, Nt_offset = 0):
        return self.env[Nt_offset:]
    
    def get_grad_envelop(self, Nt_offset = 0):
        return self.grad_env[Nt_offset:]
    
    def get_ip_pulse(self, Nt_offset = 0):
        return self.ip_pulse[Nt_offset:]
    
    def get_grad_ip(self, Nt_offset = 0):
        return self.grad_ip[Nt_offset:]
        

if __name__ == '__main__':
    Nt = 200
    fS = 80*10**6 #[Hz], sampling freq.
    fC = 5*10**6 #[Hz], carrier freq. of the pulse
    alpha = 20*10**12 #[Hz]**2
    c0 = 6300 #[m/S]
    dxdata = 0.5*10**-3 #[m]
    dzdata = 0.5* c0/(fS)
    p_def = np.array([10*dxdata, 91*dzdata])
    p_scan = np.array([5*dxdata, 0*dzdata])
    # Calculate ToF and its derivative w.r.t. x_scan
    tau = 2*np.linalg.norm(p_def - p_scan)/ c0
    grad_tau = 4* (p_def[0] - p_scan[0])/ (tau* c0**2)

    # Pulse
    pformer = PulseFormerGabor(Nt, fS, fC, alpha)
    pformer.calculate_complex_pulse(tau)
    pformer.calculate_ip_pulse(tau)
    comp_pulse = pformer.get_complex_pulse()
    ip_pulse = pformer.get_ip_pulse()
    # Derivative
    pformer.calculate_grad_complex(grad_tau)
    pformer.calculate_grad_ip(grad_tau)
    grad_comp = pformer.get_grad_complex()
    grad_ip = pformer.get_grad_ip()

    plt.figure(1)
    plt.plot(abs(comp_pulse), label = 'Magnitude of complex pulse')
    plt.plot(abs(grad_comp*0.1*10**-3), lable = 'Derivative')
    plt.title('Cpmple pulse and its dereivative w.r.t. the scan position')

    plt.figure(2)
    plt.plot(ip_pulse, label = 'IP pulse')
    plt.plot(grad_ip*0.1*10**-3, lable = 'Derivative')
    plt.title('IP pulse and its dereivative w.r.t. the scan position')
