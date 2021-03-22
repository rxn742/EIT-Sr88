#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 00:53:45 2021

@author: robgc
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "../Spyder")
from EIT_Ladder import FWHM, contrast
cwd = os.getcwd()
path = cwd + "/ProbeDiamSinglet/"

det = np.genfromtxt(path + "detunings_pion2.csv", delimiter=",")
diameters = np.linspace(0.1e-3, 7e-3, 9)
Flist = []
Conlist = []

""" 3D plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Probe beam diameter (mm)")
ax2.set_zlabel("Relative Probe Transmission")
plt.title("Effect of transit time on EIT spectrum")
for i in diameters:
    diams = np.empty(det.size)
    diams.fill(i/1e-3)
    trans = np.genfromtxt(path + f"pdiam_p_{i/1e-3:.1f}mm.csv", delimiter=",")
    #back = np.genfromtxt(path + f"pdiam_t_back{i/1e-3:.1f}mm.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    tCon = contrast(det, trans)
    Flist.append(tFWHM)
    Conlist.append(tCon)
    ax2.plot(det/(1e6), diams, trans, label= f"FWHM = {tFWHM:.2f}")
#ax2.legend(bbox_to_anchor=(-0.4, 1), loc='upper left')
ax2.view_init(30, -45)

""" FWHM against Probe Rabi """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.scatter(diameters/1e-3, Flist, color="b")
ax3.set_xlabel("Probe beam diameter (mm)")
ax3.set_ylabel("EIT FWHM (MHz)", color="b")
ax3.tick_params(axis='y', labelcolor="b")
ax4 = ax3.twinx()
ax4.scatter(diameters/1e-3, Conlist, color="r")
ax4.set_ylabel("EIT Contrast", color="r")
ax4.tick_params(axis='y', labelcolor="r")
plt.title("Effect of transit time on EIT FWHM")
fig3.tight_layout() 