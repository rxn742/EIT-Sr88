#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:28:26 2021

@author: robgc
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

defect = 3.30778
n = 56

def nstar3(x):
    return (x-defect)**3

def nfunc(x):
    return x**(1/3) + defect
    

t = np.array([1.83, 2.37, 2.73, 3.02, 3.36, 7.5])
t_err = np.array([0.2, 0.33, 0.49, 0.54, 0.71, 4.40])
nstar = np.array([15.631, 16.632, 17.633, 18.635, 19.636, 31.655])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(nstar**3, t, color='m')
ax.errorbar(nstar**3, t, t_err, color='m', linestyle='None')

def f(x, a, b):
    return a*x + b

fit = curve_fit(f, nstar**3, t, sigma=t_err)

t_56 = fit[0][0]*(n-defect)**3 + fit[0][1]

print(fit[0])

nlist = np.linspace(400, 32e3, 10000)
fitlist = f(nlist, fit[0][0], fit[0][1])

ax.plot(nlist, fitlist, color='g', label = fr"$\tau$ = {fit[0][0]:.2}$n*^3$ + {fit[0][1]:.1f}")
plt.title("Strontium 88 $^3S_1$ Triplet Rydberg Series - S. Kunze et al. (1993)")
ax.set_xlabel("Effective Quantum Number $n*^3$")
ax.set_ylabel("State Lifetime / $\mu s$")
ax.legend(loc="upper left")
lab = ["n=19","n=20","n=21","n=22","n=23","n=35"]
for i, txt in enumerate(lab):
    ax.annotate(txt, ((nstar**3)[i], t[i]),((nstar**3)[i]-1000, t[i]+1))

t_56 = fit[0][0]*(n-defect)**3 + fit[0][1]