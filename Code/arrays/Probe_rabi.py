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
path = cwd + "/arrays/ProbeRabiPop/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
rabs = np.linspace(2, 40, 20)
Flist = []
Conlist = []

""" 3D plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Probe Rabi frequency $\Omega_p$ (MHz)")
#ax2.set_zlabel("Relative Probe Transmission")
#plt.title("Effect of probe Rabi frequency on EIT Spectrum")
ax2.set_zlabel("Rydberg State Population")
plt.title("Effect of coupling Rabi frequency on Rydberg Population")
for i in rabs:
    rabi = np.empty(det.size)
    rabi.fill(i)
    trans = np.genfromtxt(path + f"probe_{i}MHz.csv", delimiter=",")
    #back = np.genfromtxt(path + f"probe_back{i}MHz.csv", delimiter=",")
    #tFWHM = FWHM(det, trans)
    tCon = contrast(det, trans)
    #Flist.append(tFWHM)
    Conlist.append(tCon)
    ax2.plot(det/(1e6), rabi, trans)#, label= f"FWHM = {tFWHM:.2f} MHz")
#ax2.legend(bbox_to_anchor=(-0.4, 1), loc='upper left')
ax2.view_init(20, -45)

""" FWHM against Probe Rabi """

#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#ax3.scatter(rabs, Flist, color="b", marker="x")
#plt.title("Effect of probe Rabi frequency on EIT FWHM")
#ax3.set_xlabel("Probe Rabi frequency $\Omega_p$ (MHz)")
#ax3.set_ylabel("EIT FWHM (MHz)", color="b")
#ax3.tick_params(axis='y', labelcolor="b")
#ax4 = ax3.twinx()
#ax4.scatter(rabs, Conlist, color="r", marker="x")
#ax4.set_ylabel("EIT Contrast", color="r")
#ax4.tick_params(axis='y', labelcolor="r")
#fig3.tight_layout()

fig3 = plt.figure()
plt.title("Effect of coupling Rabi frequency on Rydberg Population")
ax4 = fig3.add_subplot(111)
ax4.plot(rabs, Conlist, color="r")
ax4.set_xlabel("Probe Rabi frequency $\Omega_c$ (MHz)")
ax4.set_ylabel("Rydberg State Population") 