import numpy as np
import matplotlib.pyplot as plt
import time

from ultrasonic_imaging_python.definitions import units
ureg = units.ureg
Q_ = ureg.Quantity
from ultrasonic_imaging_python.sako_tools.image_quality_analyzer import ImageQualityAnalyzerMSE

"""
Modification of Jan's code
"""
plt.close('all')


def pulse(t, f_c, alpha):
    "Gaussian windowed cosine pulse"
    return np.exp(-alpha * t**2) * np.cos(2 * np.pi * f_c * t)


def p_dot(t, f_c, alpha):
    "Derivative of the pulse with regard to t"
    return -2 * alpha * t * np.exp(-alpha*t**2) * np.cos(2 * np.pi * f_c * t) -\
        2 * np.pi * f_c * np.sin(2 * np.pi * f_c * t) * np.exp(-alpha*t**2)


def tau(c0, x_s, y_s, x):
    "Hyperboloid function"
    return 2/c0 * np.sqrt((x_s - x)**2 + y_s**2)


def dtaudx(c0, x_s, y_s, x):
    "Derivative of tau with regard to the transducer position"
    return 2/c0 * (x - x_s)/(np.sqrt((x_s - x)**2 + y_s**2))


def minimize_tracking_error(a_scan_measured, tracked_position, N_it):
    a_scan_modeled = pulse((t-tau(c0, x_s, y_s, x_track)), f_c, alpha)[:, np.newaxis]
    curr_x_track = tracked_position
    curr_grad = np.zeros((a_scan_measured.shape))
    for n in range(N_it):
        curr_grad[:, 0] = 1 * dtaudx(c0, x_s, y_s, curr_x_track) * p_dot(t-tau(c0, x_s, y_s, curr_x_track), f_c, alpha)
        # np.linalg.lstsq(a,b) is the numpy equivalent to the Matlab syntax a\b to solve b= ax for x minimizing ||b - ax||_2
        tracking_error_est = np.linalg.lstsq(curr_grad,
                                             (a_scan_measured - a_scan_modeled))[0]

        a_scan_modeled = a_scan_modeled + curr_grad * tracking_error_est
        curr_x_track = curr_x_track - tracking_error_est

        err = np.linalg.norm(a_scan_measured - a_scan_modeled)
        if err <= 1e-5:
#            print("I neede so many iterations:", n)
            break
#    print(" I used all %d iterations" % N_it)
    print("Final error:", err)
    return curr_x_track, err




N_t = 500
N_offsets = 501
f_c = 2e6
alpha = 1.5e6**2
c0 = 5920
f_s = 20e6
wave_length = c0/f_c

# error between the actual transducer position and the tracked position ranging from 0 to 2 wavelength
tracking_errors = np.arange(0, N_offsets) * 2 * wave_length/(N_offsets-1)

# defect position
y_s = 250 / f_s * (c0/2)
x_s = 6e-3
# transducer position --> (x_measurement, 0)
x_measurement = 5e-3
t = np.arange(0, N_t)/f_s

# The actual measured A-scan
a_scan_measured = pulse((t-tau(c0, x_s, y_s, x_measurement)), f_c, alpha)[:, np.newaxis]
# Set analyzer
analyzer = ImageQualityAnalyzerMSE(np.array([a_scan_measured]))


error_measured_to_tracking = np.zeros((N_offsets,))
error_measured_to_updated = np.zeros((N_offsets,))
error_measured_to_updated_single_step = np.zeros((N_offsets,))
error_measured_to_updated_single_step_b = np.zeros((N_offsets,))

estimated_measurement_position = np.zeros((N_offsets,))
# for all offsets,
# 1)  calculate the modeled a-scan assuming the (wrong) tracked position
# 2) try to correct it:
#   a)Update is a first order approximation: y_model + dy_model/dx * Delta
#   b) Update is a first order approximation as y_model + dy_model/d(t-tau) * (tau(x_measured) - tau(x_tracked))
# 3) Compute the error between measured and modeled a-scan
# 4) Compute the error between measured and updated modeled a-scan
# 5) Compute the least-squares estimate of the tracking error from our approximation
for n in range(N_offsets):
    print(n)
    x_track = x_measurement + tracking_errors[n]
    # 1)
    a_scan_model = pulse((t-tau(c0, x_s, y_s, x_track)), f_c, alpha)[:, np.newaxis]
    # 2)a)
    gradient_a_scan = 1 * dtaudx(c0, x_s, y_s, x_track) * p_dot(t-tau(c0, x_s, y_s, x_track), f_c, alpha)[:, np.newaxis]
    a_scan_taylor_x = a_scan_model + gradient_a_scan * tracking_errors[n]
    # 2b)
    gradient_a_scan_b = p_dot(t-tau(c0, x_s, y_s, x_track), f_c, alpha)[:, np.newaxis]
    a_scan_taylor_tau = a_scan_model + gradient_a_scan_b * (tau(c0, x_s, y_s, x_track)-tau(c0, x_s, y_s, x_measurement))
    # 3)
    error_measured_to_tracking[n] = analyzer.get_mse(np.array([a_scan_model]))#np.linalg.norm(a_scan_measured - a_scan_model)
    # 4)
    error_measured_to_updated_single_step[n] = analyzer.get_mse(np.array([a_scan_taylor_x]))#np.linalg.norm(a_scan_measured - a_scan_taylor_x)
    error_measured_to_updated_single_step_b[n] = analyzer.get_mse(np.array([a_scan_taylor_tau]))#np.linalg.norm(a_scan_measured - a_scan_taylor_tau)
    # 5)
    start = time.time()
    estimated_measurement_position[n], error_measured_to_updated[n] =\
        minimize_tracking_error(a_scan_measured, x_track, 20)
    stop = time.time()-start
    print(stop)

plt.figure(1)
plt.clf()
plt.plot(t, a_scan_measured, label='actual measurement')
plt.plot(t, a_scan_model, label='model from tracked position')
plt.plot(t, a_scan_taylor_x, label='$+ (df/dx) \cdot (x-x_m)$')
plt.plot(t, a_scan_taylor_tau, label='$+  (df/d(t-tau(x))) \cdot (tau(x) - tau(x_m))$')
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')

plt.figure(2)
plt.clf()
plt.title('Gradient')
plt.plot(t, gradient_a_scan * tracking_errors[-1])

plt.figure(3)
plt.clf()
plt.title("Squared errors")
plt.plot(tracking_errors/wave_length, error_measured_to_tracking, label='actual tracking error')
plt.plot(tracking_errors/wave_length, error_measured_to_updated_single_step, '--', label='$+ (df/dx) \cdot (x-x_m)$')
plt.plot(tracking_errors/wave_length, error_measured_to_updated_single_step_b, '--', label='$+  (df/d(t-tau(x))) \cdot (tau(x) - tau(x_m))$')
#plt.plot(tracking_errors/wave_length, error_measured_to_updated, '--',  label='iterative')
plt.xlabel('tracking offset/wavelength')
plt.ylabel('$ || \mathbf{y} - \hat \mathbf{y} ||_2 $')
plt.legend()

plt.figure(4)
plt.clf()
plt.title('Gain')
plt.plot(tracking_errors/wave_length, error_measured_to_tracking-error_measured_to_tracking, label='no correction')
plt.plot(tracking_errors/wave_length, error_measured_to_tracking - error_measured_to_updated_single_step, label='single_step')
plt.plot(tracking_errors/wave_length, error_measured_to_tracking - error_measured_to_updated_single_step_b, label='single_step_b')
plt.plot(tracking_errors/wave_length, error_measured_to_tracking - error_measured_to_updated, label='iterative')
plt.xlabel('tracking offset/wavelength')
plt.ylabel('$ e_{tracking} - e_{ corrected}$')
plt.legend()
plt.tight_layout()


plt.figure(5)
plt.clf()
plt.title("tracking error vs estimate")
plt.plot(tracking_errors/wave_length, estimated_measurement_position, label='residual tracking error')
plt.legend()
plt.ylabel('[m]')
plt.xlabel('tracking offset/wavelength')


sample = 201
plt.figure(6)
plt.clf()
plt.plot(tracking_errors, pulse(t[sample] - tau(c0, x_s, y_s, tracking_errors), f_c, alpha),
         label='$p(250 t_0 - tau(x_ {track}))$')

plt.plot(tracking_errors, pulse((t[sample]-tau(c0, x_s, y_s, 0)), f_c, alpha) -
         dtaudx(c0, x_s, y_s, 0) *
         p_dot(t[sample]-tau(c0, x_s, y_s, 0), f_c, alpha)*tracking_errors,
         label='$f(x_{track}) + (df/dx) \cdot (x_ {track}-x_m)$')

plt.plot(tracking_errors, pulse((t[sample] - tau(c0, x_s, y_s, 0)), f_c, alpha)
         -p_dot(t[sample]-tau(c0, x_s, y_s, 0), f_c, alpha) *
         (tau(c0, x_s, y_s, tracking_errors)-tau(c0, x_s, y_s, 0)),
         label='$ (df/d(t-tau(x))) \cdot (tau(x_ {track}) - tau(x_m))$')
plt.legend()
plt.xlabel('tracking_error[m]')
#