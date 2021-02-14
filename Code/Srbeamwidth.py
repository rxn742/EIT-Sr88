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
path = cwd + "/arrays/SrWidth/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
sls = np.linspace(1e-3, 20e-3, 20)
Flist = []

""" 3D plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Atomic Beam width $l$ ($mm$)")
ax2.set_zlabel("Relative Probe Transmission")
plt.title("Effect of Atomic Beam Width on EIT spectrum")
for i in sls:
    sl = np.empty(det.size)
    sl.fill(i/1e-3)
    trans = np.genfromtxt(path + f"beamwidth_{i/1e-3:.1f}mm.csv", delimiter=",")
    back = np.genfromtxt(path + f"beamwidth_back{i/1e-3:.1f}mm.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    Flist.append(tFWHM)
    ax2.plot(det/(1e6), sl, trans, label= f"FWHM = {tFWHM:.2f}")
ax2.legend()

""" FWHM against Probe Rabi """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(sls/1e-3, Flist)
plt.title("Effect of Atomic Beam Width on EIT FWHM")
ax3.set_xlabel("Atomic Beam width $l$ ($mm$)")
ax3.set_ylabel("EIT FWHM (MHz)")