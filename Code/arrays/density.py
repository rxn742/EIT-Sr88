#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 00:53:45 2021

@author: robgc
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from EIT_Ladder import FWHM, contrast
cwd = os.getcwd()
path = cwd + "/arrays/DensityP+40/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
densities = np.linspace(1e13, 2e15, 21)
Flist = []
Conlist = []

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
    trans = np.genfromtxt(path + f"density5_{i/1e15:.2f}MHz.csv", delimiter=",")
    back = np.genfromtxt(path + f"density5_back{i/1e15:.2f}MHz.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    tCon = contrast(det, trans)
    Flist.append(tFWHM)
    Conlist.append(tCon)
    ax2.plot(det/(1e6), density, trans-back, label= f"FWHM = {tFWHM:.2f}")
ax2.legend(bbox_to_anchor=(-0.4, 1), loc='upper left')
ax2.view_init(30, -45)

""" FWHM against Probe Rabi """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(densities/1e15, Flist, color="b")
ax3.set_xlabel("Coupling Rabi frequency $\Omega_c$ (MHz)")
ax3.set_ylabel("EIT FWHM (MHz)", color="b")
ax3.tick_params(axis='y', labelcolor="b")
ax4 = ax3.twinx()
ax4.plot(densities/1e15, Conlist, color="r")
ax4.set_ylabel("EIT Contrast", color="r")
ax4.tick_params(axis='y', labelcolor="r")
plt.title("Effect of Atomic Density on EIT FWHM")
ax3.set_xlabel("Atomic Density $n$ ($10^9 cm^-3$)")
ax3.set_ylabel("EIT FWHM (MHz)", color="b")
fig3.tight_layout() 