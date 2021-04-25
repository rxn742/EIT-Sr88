from backend import pop_calc
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from multiprocessing import set_start_method

def Rydberg_pop(omega_p, delta_c, omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, gauss, temperature, kp, kc, beamdiv, probe_diameter, coupling_diameter, tt):

	dlist, plist = pop_calc(delta_c, omega_p, omega_c, spontaneous_32, 
                            spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, (2,2), 
                            gauss, temperature, kp, kc, beamdiv, probe_diameter, coupling_diameter, tt)
	peaks = find_peaks(plist)[0]
	vals = plist[peaks]
	opt = np.max(vals)
	return -opt

def optimise(omega_p, delta_c, omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, gauss, temperature, kp, kc, beamdiv, probe_diameter, coupling_diameter, tt):
	optimal = minimize(Rydberg_pop, omega_p, args=(delta_c, omega_c, spontaneous_32, spontaneous_21, lw_probe, lw_coupling, dmin, dmax, steps, gauss, temperature, kp, kc, beamdiv, probe_diameter, coupling_diameter, tt), method='Nelder-Mead')
	print(optimal)










if __name__ == "__main__":
	set_start_method("spawn")
	
	neg = optimise(5e6, 0, 15e6, 4e4, 32e6*2*np.pi, 1e6, 1e6, -314e6, 314e6, 10000, "N", 623.15, 2*np.pi/461e-9, 2*np.pi/461e-9, 38e-3, 1, 1, "N")
	print(neg)











