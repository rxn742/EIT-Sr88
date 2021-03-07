#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:28:26 2021

@author: robgc
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import hbar, e, epsilon_0,c, m_e
from sys import path
path.insert(0, "../GUI")
from vals_413 import d12_413, d23_413
from vals_318 import d12_318, d23_318

def nstar3(x, defect):
    return (x-defect)**3

def nfunc(x, defect):
    return x**(1/3) + defect

def f(x, a, b):
    return a*x + b
    
def state_1S0(n):
    defect = 3.269
    fit = curve_fit(f, nstar**3, t, sigma=t_err)
    t_n = fit[0][0]*(n-defect)**3 + fit[0][1]
    return f"{t_n:.2e}"

def state_1D2(n):
    defect = 2.381
    fit = curve_fit(f, nstar**3, t, sigma=t_err)
    t_n = fit[0][0]*(n-defect)**3 + fit[0][1]
    return f"{t_n:.2e}"

def state_3S0(n):
    defect = 3.371
    t = np.array([1.83, 2.37, 2.73, 3.02, 3.36, 7.5])
    t_err = np.array([0.2, 0.33, 0.49, 0.54, 0.71, 4.40])
    nstar = np.array([15.631, 16.632, 17.633, 18.635, 19.636, 31.655])
    fit = curve_fit(f, nstar**3, t, sigma=t_err)
    t_n = fit[0][0]*(n-defect)**3 + fit[0][1]
    return f"{t_n:.2f} micro seconds"

def state_3D3(n):
    defect = 2.630
    t = np.array([0.77, 0.94, 1.13, 1.33, 1.58, 1.77])
    t_err = np.array([0.03, 0.05, 0.08, 0.1, 0.14, 0.19])
    nstar = np.array([20.586, 21.559, 22.536, 23.521, 24.510, 25.499])
    fit = curve_fit(f, nstar**3, t, sigma=t_err)
    t_n = fit[0][0]*(n-defect)**3 + fit[0][1]
    return f"{t_n:.2f} micro seconds"

#fig = plt.figure()
#ax = fig.add_subplot(111)
#nlist = np.linspace(400, 32e3, 10000)
#fitlist = f(nlist, fit[0][0], fit[0][1])
#ax.plot(nlist, fitlist, color='g', label = fr"$\tau$ = {fit[0][0]:.2}$n*^3$ + {fit[0][1]:.1f}")
#plt.title("Strontium 88 $^3S_1$ Triplet Rydberg Series - S. Kunze et al. (1993)")
#ax.set_xlabel("Effective Quantum Number $n*^3$")
#ax.set_ylabel("State Lifetime / $\mu s$")
#ax.legend(loc="upper left")
#lab = ["n=19","n=20","n=21","n=22","n=23","n=35"]
#for i, txt in enumerate(lab):
#    ax.annotate(txt, ((nstar**3)[i], t[i]),((nstar**3)[i]-1000, t[i]+1))
#ax.scatter(nstar**3, t, color='m')
#ax.errorbar(nstar**3, t, t_err, color='m', linestyle='None')
