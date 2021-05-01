import numpy as np
import ray
from scipy.signal import find_peaks
from backend_ray import pop_calc, tcalc, FWHM, contrast
from copy import deepcopy
import time

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
        self.constants = ["spontaneous_21", "spontaneous_32", "kp", "kc", "dig", "dri"]
        self.class_kwargs = kwargs

    def EIT(self, **kwargs):
        required_vars = list(tcalc.__code__.co_varnames[:tcalc.__code__.co_argcount])
        dic = {**self.class_kwargs, **kwargs}
        
        for i in required_vars:
            if i not in dic.keys():
                if i not in self.constants:
                        raise Error(f"Keyword argumemt {i} has not been defined")
        
        dlist, tlist = tcalc(dic["delta_c"], dic["omega_p"], dic["omega_c"], self.spontaneous_32, 
                                    self.spontaneous_21, dic["lw_probe"], dic["lw_coupling"], dic["dmin"], 
                                    dic["dmax"], dic["steps"], dic["gauss"], self.kp, self.kc, dic['density'], 
                                    self.d12, dic['sl'], dic["temperature"], dic["beamdiv"], 
                                    dic["probe_diameter"], dic["coupling_diameter"], dic["tt"])
        
        try:
            lw = FWHM(dlist, tlist)
        except:
            lw = 0
        try:
            ct = contrast(dlist, tlist)
        except:
            ct = 0
        return lw, ct
                        
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
        
    def exploration(self, points):
        """
        This function forms part of the patern search algorithm, taking a start point and 
        evaluating the metric in the near vicinity of the point to check if those are closer
        to the maxima.
        
        Inputs:
            start_points - locations of the initail 2 variable cut points, type = tuple
            step_size - The size of the perturbation from the start_points to explore, type = float
        Output:
            Improved values for cut locations, type = tuple
        """
        perturbation = deepcopy(points)
        new_points = {}
        pop = -self.Rydberg_pop(**points)
        for i in points:
            perturbation[i] = points[i]+self.step
            new_pop_plus = -self.Rydberg_pop(**perturbation)
            if new_pop_plus < pop:
                new_points[i] = perturbation[i]
            else:
                perturbation[i] = points[i]-2*self.step
                new_pop_minus = -self.Rydberg_pop(**perturbation)
                if new_pop_minus < pop:
                    new_points[i] = perturbation[i]
                else:
                    new_points[i] = points[i]
        return new_points
                    
    def pattern_search(self, step_size=1e6, **kwargs):
        """
        This function performs the pattern search algorithm to find the maximum value of
        the significance function.
        
        Inputs:
            start_points - locations of the initail 2 variable cut points, type = tuple
            step - The size of the perturbation from the start_points to explore, type = float
        Output:
            The optimised cut values in two dimensions.
        """
        
        keys = list(self.variables.keys())
        for i in keys:
            if i not in kwargs.keys():
                raise Error("All parameters require a starting value")
    
        self.step = step_size
        self.points = kwargs
        self.test_points = {}
        self.vector = {}
    
        c = 0
        while self.step > 10:
            print(self.points)
            c += 1
            f = -self.Rydberg_pop(**self.points)
            print(f"f = {f}")
            self.test_points = self.exploration(self.points)
            fe = -self.Rydberg_pop(**self.test_points)
            print(f"fe ={fe}")
            if fe < f:
                print("Explored value is better")                
                for i in self.points:
                    self.vector[i] = self.points[i]+2*(self.test_points[i]-self.points[i])
                self.pattern_points = self.exploration(self.vector)
                fp = -self.Rydberg_pop(**self.pattern_points)
                print(f"fp pattern step = {fp}")
                if fp < fe:
                    self.points = self.pattern_points
                else:
                    self.points = self.test_points
            if self.points == self.test_points:
                self.step = self.step/10 # reduces the step size
                self.points = self.exploration(self.points)
        print(f)
        return f

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
        
        while b-a > 100:
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

    def cm_init(self, EIT, **kwargs):
        self.linewidth = 2
        keys = list(self.variables.keys())
        for i in keys:
            if i not in kwargs.keys():
                raise Error("All parameters require a starting value")
        self.x1 = kwargs
        if self.cm_max(**self.x1) and self.cm_constraints(**self.x1):
            count = 1
        else:
            raise Error("Original point does not meet the contstraints")
        self.xc = {"1":self.x1}
        self.fns = {"1":self.EIT(**kwargs)[0]}
        
        while count < 4:
            self.xi = {}
            count+=1
            feasible = False
            while feasible == False:    
                for j in self.x1:
                    self.xi[j] = np.random.uniform()*self.maximums[j]
                
                if self.cm_constraints(**self.xi):
                    feasible = True
                else:
                    for j in self.xi:
                        self.xi[j] = 1/2*(self.xi[j] + self.xc["1"][j])
                    
            self.xc[f"{count}"] = self.xi
            self.fns[f"{count}"] = self.EIT(**self.xi)[0]

        return self.xc, self.fns
    
    def cm_iter(self, EIT="no", alpha=1, **kwargs):
        cplex, F = self.cm_init(EIT, **kwargs)
        std_dev = 1
        while std_dev > 1e-4:
            max_key = max(F, key=self.fns.get)
            xh = cplex.pop(max_key)
            fn_max = F.pop(max_key)
            oc_centroid = np.array([cplex[i]["omega_c"] for i in cplex]).mean()
            op_centroid = np.array([cplex[i]["omega_p"] for i in cplex]).mean()
            xo = {"omega_p":op_centroid, "omega_c":oc_centroid}
            xr = {}
            for k in xh:
                xr[k] = (1+alpha)*xo[k] - alpha*xh[k]
            feasible = False
            while feasible == False:
                if self.cm_max(**xr):
                    if self.cm_constraints(**xr):
                        fn_reflect = self.EIT(**xr)[0]
                        if fn_reflect < fn_max:
                            feasible = True
                            xh = xr
                            cplex[max_key] = xh
                            F[max_key] = fn_reflect
                        else:
                            for j in xr:
                                xr[j] = 1/2*(xo[j]+xr[j])
                    else:
                        for j in xr:
                            xr[j] = 1/2*(xo[j]+xr[j])   
                else:
                    for j in xr:
                        if self.maximums[j] < xr[j]:
                            xr[j] = self.maximums[j] - 1e-6
                    if self.cm_constraints(**xr):
                        for j in xr:
                            xr[j] = 1/2*(xo[j]+xr[j])
            
            std_dev = np.array([F[i] for i in F]).std()
            vals = list(cplex.values())
            fvals = list(F.values())
            print(f"Function = {fvals}")
            print("\n")
            print(f"vals = {vals}")
            print("\n")
        
        min_key = min(F, key=self.fns.get)
        xe = cplex.pop(min_key)
        fn_min = self.EIT(**xe)[0]
        print(f'Rydberg population = {fn_min*100:.2f}% \nOmega_p = {xe["omega_p"]} Hz \nOmega_c = {xe["omega_c"]} Hz')
        return [fn_min, xe["omega_p"], xe["omega_c"]]

    def cm_max(self, **kwargs):
        self.maximums = {}
        self.minimums = {}
        self.maximums["omega_p"] = 15e6
        self.maximums["omega_c"] = 15e6
        self.minimums["omega_p"] = 0
        self.minimums["omega_c"] = 0
        if all(self.maximums[j] > kwargs[j] for j in kwargs) and all(self.minimums[j] < kwargs[j] for j in kwargs):
            return True
        else:
            return False
    
    def cm_constraints(self, **kwargs):
        self.contrast = 0.01
        #print(self.EIT(**kwargs)[0]/(2*np.pi*1e6))
        if self.EIT(**kwargs)[1] > self.contrast and kwargs["omega_c"] >= kwargs["omega_p"]:
            return True
        else:
            return False
        
        
        

if __name__ == "__main__":
    #ray.shutdown()
    try:
        ray.init(address='auto', _redis_password='5241590000000000')
    except:
        ray.init()
        pass
    start = time.time()
    #result = bisection(1e6, 40e6, delta_c=0, omega_p=10e6, lw_probe=1e5, lw_coupling=1e5, dmin=-100e6, dmax=100e6, steps=1000, gauss="N", temperature=623.15, beamdiv=38e-3, probe_diameter=1, coupling_diameter=1, tt="N")
    #print(result)
    #ray.shutdown()
    op = optimise("singlet", "Rydberg", "omega_c", delta_c=0, omega_p=40e6, lw_probe=1e6, lw_coupling=1e6, dmin=-60e6, dmax=60e6, steps=100, gauss="Y", temperature=650, beamdiv=38e-3, probe_diameter=4e-3, coupling_diameter=1.69e-3, tt="Y", density=1e15, sl=3e-3)
    #op.bisection1D(1e6, 30e6)
    #op.pattern_search(omega_c=15e6, omega_p=5e6)
    vals = op.cm_iter(omega_c=10e6, omega_p=1e6)
    time = {time.time()-start}
    print(time)
    with open("EIT.txt", "a") as f:
        for item in vals:
            f.write(str(item) + "\n")








