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
path = cwd + "/arrays/CounterPropagating/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
sigs = np.linspace(5, 100, 20)
nodb = np.genfromtxt(path + "no_db.csv", delimiter=",")
nodb_back = np.genfromtxt(path + "nodb_background.csv", delimiter=",")
nodbFWHM = FWHM(det, nodb)
Flist = []
Flist.append(nodbFWHM)

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
ax2.plot(det/(1e6), np.zeros(det.size), nodb-nodb_back, label= f"FWHM = {nodbFWHM:.2f}")
ax2.set_xlabel("$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Doppler width $\sigma_v$")
ax2.set_zlabel("Relative Probe Transmission")
plt.title("Effect of Doppler width on EIT Spectrum")
for i in sigs:
    dv = np.empty(det.size)
    dv.fill(i)
    trans = np.genfromtxt(path + f"db_{i}.csv", delimiter=",")
    back = np.genfromtxt(path + f"count_back_{i}.csv", delimiter=",")
    tFWHM = FWHM(det, trans)
    Flist.append(tFWHM)
    ax2.plot(det/(1e6), dv, trans-back, label= f"FWHM = {tFWHM:.2f}")
ax2.legend()

""" FWHM against Doppler width """

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sigs2 = np.linspace(0, 100, 21)
ax3.plot(sigs2, Flist)
plt.title("Effect of Doppler width on EIT FWHM")
ax3.set_xlabel("Doppler width $\sigma_v$")
ax3.set_ylabel("EIT FWHM (MHz)")