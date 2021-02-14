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
path = cwd + "/arrays/Density/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
densities = np.linspace(1e13, 2e15, 21)
Flist = []

""" 3D plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Atomic Density $n$ ($10^9 cm^-3$)")
ax2.set_zlabel("Relative Probe Transmission")
plt.title("Effect of Atomic Density on EIT spectrum")
for i in densities:
    density = np.empty(det.size)
    density.fill(i/1e15)
    trans = np.genfromtxt(path + f"density_{i/1e15:.2f}MHz.csv", delimiter=",")
    back = np.genfromtxt(path + f"density_back{i/1e15:.2f}MHz.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    Flist.append(tFWHM)
    ax2.plot(det/(1e6), density, trans-back, label= f"FWHM = {tFWHM:.2f}")
ax2.legend()

""" FWHM against Probe Rabi """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(densities/1e15, Flist)
plt.title("Effect of Atomic Density on EIT FWHM")
ax3.set_xlabel("Atomic Density $n$ ($10^9 cm^-3$)")
ax3.set_ylabel("EIT FWHM (MHz)")