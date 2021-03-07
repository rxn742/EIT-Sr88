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
path = cwd + "/arrays/CounterPropagating/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
sigs = np.linspace(5, 100, 20)
#sigs = sigs[:-8]
nodb = np.genfromtxt(path + "no_db.csv", delimiter=",")
nodb_back = np.genfromtxt(path + "nodb_background.csv", delimiter=",")
nodbFWHM = FWHM(det, nodb)
nodbCon = contrast(det, nodb)
Flist = []
Conlist = []
Flist.append(nodbFWHM)
Conlist.append(nodbCon)

""" 2D Plot """

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(det/(1e6), nodb, label= f"$\sigma_v = 0 m/s$, FWHM = {nodbFWHM:.2f}")
#ax.set_xlabel(r"$\Delta_p$ ($MHz$)")
#ax.set_ylabel(r"Probe Transmission")
#plt.title("Effect of Doppler width on EIT Spectrum")
#for i in sigs:
#    trans = np.genfromtxt(path + f"db_{i}.csv", delimiter=",")
#    tFWHM = FWHM(det, trans)
#    Flist.append(tFWHM)
#    ax.plot(det/(1e6), trans, label= rf"$\sigma_v = {i:.0f} m/s$, FWHM = {tFWHM:.2f}")
#ax.legend()

""" 3D plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")
ax2.plot(det/(1e6), np.zeros(det.size), nodb-nodb_back, label= f"FWHM = {nodbFWHM:.2f} MHz")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel(r"Velocity Distribution width ($m/s$)")
ax2.set_zlabel("Relative Probe Transmission")
plt.title("Effect of Doppler broadening on EIT Spectrum")
for i in sigs:
    dv = np.empty(det.size)
    dv.fill(i)
    trans = np.genfromtxt(path + f"db_{i}.csv", delimiter=",")
    back = np.genfromtxt(path + f"count_back_{i}.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    tCon = contrast(det, trans)
    Flist.append(tFWHM)
    Conlist.append(tCon)
    ax2.plot(det/(1e6), dv/np.sqrt(3), trans-back, label= f"FWHM = {tFWHM:.2f} MHz")
ax2.legend(bbox_to_anchor=(-0.4, 1), loc='upper left')
ax2.view_init(20, -45)

""" FWHM against Doppler width """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sigs2 = np.linspace(0, 100, 21)
ax3.scatter(sigs2/np.sqrt(3), Flist, color="b", marker="x")
plt.title("Effect of Doppler broadening on EIT Peak")
ax3.set_xlabel(r"Velocity Distribution Width ($m/s$)")
ax3.set_ylabel("EIT FWHM (MHz)", color="b")
ax3.tick_params(axis='y', labelcolor="b")
ax4 = ax3.twinx()
ax4.scatter(sigs2/np.sqrt(3), Conlist, color="r", marker="x")
ax4.set_ylabel("EIT Contrast", color="r")
ax4.tick_params(axis='y', labelcolor="r")
fig3.tight_layout() 