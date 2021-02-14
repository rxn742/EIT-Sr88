#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 00:53:45 2021

@author: robgc
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from EIT_Ladder import FWHM
cwd = os.getcwd()
path = cwd + "/arrays/CouplingRabi/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
rabs = np.linspace(2, 40, 20)
Flist = []

""" 3D plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Coupling Rabi frequency $\Omega_c$ (MHz)")
ax2.set_zlabel("Relative Probe Transmission")
plt.title("Effect of coupling Rabi frequency on EIT Spectrum")
for i in rabs:
    rabi = np.empty(det.size)
    rabi.fill(i)
    trans = np.genfromtxt(path + f"coupling_{i}MHz.csv", delimiter=",")
    back = np.genfromtxt(path + f"coupling_back{i}MHz.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    Flist.append(tFWHM)
    ax2.plot(det/(1e6), rabi, trans-back, label= f"FWHM = {tFWHM:.2f}")
ax2.legend()

""" FWHM against Probe Rabi """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(rabs, Flist)
plt.title("Effect of coupling Rabi frequency on EIT FWHM")
ax3.set_xlabel("Coupling Rabi frequency $\Omega_c$ (MHz)")
ax3.set_ylabel("EIT FWHM (MHz)")