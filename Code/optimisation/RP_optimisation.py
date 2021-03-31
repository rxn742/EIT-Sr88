import numpy as np
from sys import path
path.insert(0, "../GUI")
import ray
from scipy.signal import find_peaks
from backend_ray import pop_calc

class Error(Exception):
    """
    Class for deailing with errors that are called.
    """
    pass

class optimise:
    
    def __init__(self, system, opt, *args, **kwargs):
        
        if system == "singlet":
            from vals_413 import d12_413, d23_413, spontaneous_21_413, spontaneous_32_413, kp_413, kc_413
            self.spontaneous_21 = spontaneous_21_413
            self.spontaneous_32 = spontaneous_32_413
            self.kp = kp_413
            self.kc = kc_413
            self.d12 = d12_413
            self.d23 = d23_413
        elif system == "triplet":
            from vals_318 import d12_318, d23_318, spontaneous_21_318, spontaneous_32_318, kp_318, kc_318
            self.spontaneous_21 = spontaneous_21_318
            self.spontaneous_32 = spontaneous_32_318
            self.kp = kp_318
            self.kc = kc_318
            self.d12 = d12_318
            self.d23 = d23_318
        else:
            raise Error("Please select either singlet or triplet series")
            
        if opt == "Rydberg":
            self.opt = "Rydberg"
        elif opt == "EIT":
            self.opt = "EIT"
        else:
            raise Error("Please select either Rydberg or EIT to optimise")
            
        variables = {}
        for i in args:
            print(i)
            if type(i) == str:
                variables[i] = 0
            else:
                raise Error("Enter optimisation variables as strings")
        self.variables = variables
        self.constants = ["spontaneous_21", "spontaneous_32", "kp", "kc", "d12", "d23"]
        self.class_kwargs = kwargs

    def Rydberg_pop(self, **kwargs):
        
        required_vars = list(pop_calc.__code__.co_varnames[:pop_calc.__code__.co_argcount])
        dic = {**self.class_kwargs, **kwargs}
        
        for i in required_vars:
            if i not in dic.keys():
                if i not in self.constants:
                    if i != "state_index":
                        raise Error(f"Keyword argumemt {i} has not been defined")
        
        dlist, plist = pop_calc(dic["delta_c"], dic["omega_p"], dic["omega_c"], self.spontaneous_32, 
                                    self.spontaneous_21, dic["lw_probe"], dic["lw_coupling"], dic["dmin"], 
                                    dic["dmax"], dic["steps"], (2,2), dic["gauss"], dic["temperature"], self.kp, 
                                    self.kc, dic["beamdiv"], dic["probe_diameter"], dic["coupling_diameter"], dic["tt"])
       
        peaks = find_peaks(plist)[0]
        vals = plist[peaks]
        opt = np.max(vals)
        return opt

    def bisection1D(self, a, b):
        """
        This function implements the one dimensional bisection 
        method, either performing and upper or lower 
        cut depending on whether the signal or background has a larger mean.
        
        Input:
            a - lower limit of optimisation range, type = float
            b - upper limit of optimisation range, type = float
        Output:
            The optimised cut location to maximise significance, type = float
        """    
        
        if len(self.variables) != 1:
            raise Error("Bisection is only valid in 1D")
        key = list(self.variables.keys())[0]
        
        while b-a > 1e3:
            params = {key:a}
            f_a = self.Rydberg_pop(**params)
            params = {key:b}
            f_b = self.Rydberg_pop(**params)
            m = (a+b)/2 # midpoint calculation
            params = {key:m}
            f_m = self.Rydberg_pop(**params)
            r = (m+b)/2
            params = {key:r}
            f_r = self.Rydberg_pop(**params)
            l = (m+a)/2
            params = {key:l}
            f_l = self.Rydberg_pop(**params)
            k = [f_a, f_b, f_m, f_r, f_l]
            k_max = max(k)
            mv = k.index(k_max) # finds the maximum value of the metric
            if mv == 0:
                b = m
                print(a, k_max, b-a)
            if mv == 4:
                b = m
                print(l, k_max, b-a)
            if mv == 1:
                a = m
                print(b, k_max, b-a)
            if mv == 3:
                a = m
                print(r, k_max, b-a)            
            if mv == 2:
                a = l
                b = r
                print(m, k_max, b-a)        
        return a



if __name__ == "__main__":
    try:
        ray.init(address='auto', _redis_password='5241590000000000')
    except:
        ray.init()
    #result = bisection(1e6, 40e6, delta_c=0, omega_p=10e6, lw_probe=1e5, lw_coupling=1e5, dmin=-100e6, dmax=100e6, steps=1000, gauss="N", temperature=623.15, beamdiv=38e-3, probe_diameter=1, coupling_diameter=1, tt="N")
    #print(result)
    #ray.shutdown()
    op = optimise("singlet", "Rydberg", "omega_c", delta_c=0, omega_p=10e6, lw_probe=1e5, lw_coupling=1e5, dmin=-100e6, dmax=100e6, steps=1000, gauss="N", temperature=623.15, beamdiv=38e-3, probe_diameter=1, coupling_diameter=1, tt="N")
    op.bisection1D(1e6, 30e6)









