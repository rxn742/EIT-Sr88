#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:28:26 2021

@author: robgc
"""

import matplotlib.pyplot as plt
import numpy as np

defect = 2.63
n = 56

t = np.array([2.37, 2.73, 3.02, 3.36, 7.5])
nstar = np.array([16.63, 17.63, 18.64, 19.64, 31.66])

plt.scatter(nstar, t**(1/3))

fit = np.polyfit(nstar, t**(1/3), 1)

t_56 = ((n-defect)*fit[0]+fit[1])**3