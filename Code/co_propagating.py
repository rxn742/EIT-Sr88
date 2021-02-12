#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:57:11 2021

@author: robgc
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EIT_Ladder import FWHM
cwd = os.getcwd()
path = cwd + "/arrays/CoPropagating/"

det = np.genfromtxt(path + "detunings.csv", delimiter=",")
sigs = np.linspace(5, 100, 20)
nodb = np.genfromtxt(path + "no_db.csv", delimiter=",")
nodbFWHM = FWHM(det, nodb)
nodb_back = np.genfromtxt(path + "nodb_background.csv", delimiter=",")
wlist = np.linspace(0.1, 0.9, 9)
co_list = []

""" 2D Plot """

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(det/(1e6), nodb, label= f"$\sigma_v = 0 m/s$, FWHM = {nodbFWHM:.2f}")
#ax.set_xlabel(r"$\Delta_p$ ($MHz$)")
#ax.set_ylabel(r"Probe Transmission")
#for i in wlist:
#    co = np.genfromtxt(f"co_{i:.1f}.csv", delimiter=",")
#    try:
#        coFWHM = FWHM(det, co)
#    except:
#        coFWHM = np.inf
#    co_list.append(coFWHM)
#    ax.plot(det/(1e6), co, label= f"$\sigma_v = {i:.0f} m/s$, FWHM = {coFWHM:.2f}")
#ax.legend()
#plt.title("Co-Propagating")

""" 3D Plot """

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.plot(det/(1e6), np.zeros(det.size), nodb-nodb_back, label= f"FWHM = {nodbFWHM:.2f}")
ax2.set_xlabel(r"$\Delta_p$ ($MHz$)")
ax2.set_ylabel("Doppler width $\sigma_v$")
ax2.set_zlabel("Relative Probe Transmission")

for i in wlist:
    dv = np.empty(det.size)
    dv.fill(i)
    co = np.genfromtxt(path + f"co_{i:.1f}.csv", delimiter=",")
    co_back = np.genfromtxt(path + f"co_background{i:.1f}.csv", delimiter=",")
    try:
        coFWHM = FWHM(det, co)
        if i != wlist[0]:
            if coFWHM < co_list[-1]:
                coFWHM = np.inf
    except:
        coFWHM = np.inf
    co_list.append(coFWHM)
    ax2.plot(det/(1e6), dv, co-co_back, label= f"FWHM = {coFWHM:.2f}")
ax2.legend()
plt.title("Co-Propagating")
