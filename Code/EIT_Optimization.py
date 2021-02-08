#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:47:20 2021

@author: robgc
"""

import qutip as qt
qt.settings.auto_tidyup=False
import numpy as np
from scipy.constants import hbar,epsilon_0,c
import matplotlib.pyplot as plt
from matplotlib import cm
from EIT_413nm import *

global dri
global dig

def tcalcnod(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, dmin, dmax, steps):
    """
    This function generates an array of transmission values for a generated list of probe detunings
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in Hz.
    Omega_p : float
        Probe Rabi frequency in Hz.
    Omega_c : float
        Coupling Rabi frequency in Hz.
    gamma_ri : float
        r-i spontaneous emission rate.
    gamma_ig : float
        i-g spontaneous emission rate.
    lwp : float
        Probe beam linewidth in Hz
    lwc : float
        Coupling beam linewidth in Hz
    dmin : float
        Lower bound of Probe detuning in MHz
    dmax : float
        Upper bound of Probe detuning in MHz
    steps : int
        Number of Probe detunings to calculate the transmission at 

    Returns
    -------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    tlist : numpy.ndarray, dtype = float64
        Array of transmission values corresponding to the detunings

    """
    tlist = np.empty(steps+1)
    dlist = np.empty(steps+1)
    d=(dmax-dmin)*(steps)**(-1)
    for i in range(0, steps+1):
        dlist[i] = dmin
        tlist[i] = transmission(density, dmin, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)
        dmin+=d
    return dlist, tlist

def optimize1D(delta_c, gamma_ri, gamma_ig, lwp, lwc, points=1000):
    """
    This function calculates an optimium Rabi frequency of one laser when keeping the other constant
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in Hz.
    gamma_ri : float
        r-i spontaneous emission rate.
    gamma_ig : float
        i-g spontaneous emission rate.
    lwp : float
        Probe beam linewidth in Hz
    lwc : float
        Coupling beam linewidth in Hz
    
    Returns
    -------
    RP : plot
        Plot of maximum Rydberg state population probability against chosen Rabi frequency 
    """
    rabi = input("Which Rabi frequency do you want to change? \nProbe or Coupling \n")
    ran = input("Enter min and max values \n")
    ran = ran.split(",")
    rp = []
    if rabi == "Probe":
        Omega_p = np.linspace(float(ran[0]), float(ran[1]), points)
        Omega_c = Oc
        for i in range(len(Omega_p)):
            pop = np.abs(population(-delta_c, delta_c, Omega_p[i], Oc, gamma_ri, gamma_ig, lwp, lwc)[2,2])
            rp.append(pop)
    if rabi == "Coupling":
        Omega_c = np.linspace(float(ran[0]), float(ran[1]), points)
        Omega_p = Op
        for i in range(len(Omega_c)):
            pop = np.abs(population(-delta_c, delta_c, Op, Omega_c[i], gamma_ri, gamma_ig, lwp, lwc)[2,2])
            rp.append(pop)
    try:
        peak = find_peaks(rp)[0][0]
    except IndexError:
        pass
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(f"Rydberg state population against {rabi} Rabi frequency")
    ax.set_ylabel("Rydberg state population")
    if rabi == "Probe":
        ax.set_xlabel("$\Omega_p$ / MHz")
        ax.plot(Omega_p/1e6, rp, label = r"$\Omega_c =$" f"{Omega_c/1e6:.1f}" r"$MHz$")
    if rabi == "Coupling":
        ax.set_xlabel("$\Omega_c$ / MHz")
        ax.plot(Omega_c/1e6, rp, label = r"$\Omega_p =$" f"{Omega_p/1e6:.1f}" r"$MHz$")
        ax.scatter(Omega_c[peak]/1e6, rp[peak], label = r"Max at $\frac{\Omega_c}{\Omega_p}$ =" f"{Omega_c[peak]/Omega_p:.2f}", color="green")
    ax.legend()
    plt.show()
    
def optimize2D(delta_c, gamma_ri, gamma_ig, lwp, lwc, diameter, points=100):
    """
    This function calculates optimium Rabi frequencies to maximise Rydberg population probability 
    Parameters
    ---------- 
    delta_c : float
        Coupling detuning in Hz.
    gamma_ri : float
        r-i spontaneous emission rate.
    gamma_ig : float
        i-g spontaneous emission rate.
    lwp : float
        Probe beam linewidth in Hz
    lwc : float
        Coupling beam linewidth in Hz
    
    Returns
    -------
    RP : plot
        Surface plot of maximum Rydberg state population probability against each Rabi frequency 
    """
    opt = input("What would you like to optimise out of Rydberg population, EIT peak height and EIT peak width? \nEIT H, EIT W or Rydberg \n")
    probe_power = input("Enter probe beam power range in Watts\n")
    coupling_power = input("Enter coupling beam power ramge in Watts\n")
    probe_power = probe_power.split(",")
    coupling_power = coupling_power.split(",")
    pplist = np.linspace(float(probe_power[0]), float(probe_power[1]), points)
    cplist = np.linspace(float(coupling_power[0]), float(coupling_power[1]), points)
    Ip = pplist/(np.pi*(diameter/2)**2)
    Ic = cplist/(np.pi*(diameter/2)**2)    
    Omega_c = (dri/hbar)*np.sqrt((2*Ic)/(c*epsilon_0*1.0003)) # Coupling Rabi frequency
    Omega_p = (dig/hbar)*np.sqrt((2*Ip)/(c*epsilon_0*1.0003)) # Probe Rabi frequency
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    x, y = np.meshgrid(Omega_p, Omega_c)
    Z = np.zeros((len(y), len(x)))
    if opt == "Rydberg":
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i,j] = np.abs(population(-delta_c, delta_c, x[i,j], y[i,j], gamma_ri, gamma_ig, lwp, lwc)[2,2])
        m = ax.pcolor(x/1e6, y/1e6, Z, cmap=cm.coolwarm)
        ax.set_ylabel("$\Omega_c$ / MHz")
        ax.set_xlabel("$\Omega_p$ / MHz")
        plt.colorbar(m)
        plt.title(r"Rydberg state population probability against Rabi frequency")
        plt.show()
    if opt == "EIT H":
        for i in range(len(x)):
            print(i)
            for j in range(len(y)):
                Z[i,j] = np.abs(transmission(density, -delta_c, delta_c, x[i,j], y[i,j], gamma_ri, gamma_ig, lwp, lwc))
        m = ax.pcolor(x/1e6, y/1e6, Z, cmap=cm.coolwarm)
        ax.set_ylabel("$\Omega_c$ / MHz")
        ax.set_xlabel("$\Omega_p$ / MHz")
        plt.colorbar(m)
        plt.title(r"EIT transmission peak height against Rabi frequency")
        plt.show()
    if opt == "EIT W":
        for i in range(len(x)):
            print(i)
            for j in range(len(y)):
                dlist, tlist = tcalcnod(delta_c, x[i,j], y[i,j], gamma_ri, gamma_ig, lwp, lwc, -50e6, 50e6, 100)
                try:
                    Z[i,j] = FWHM(dlist, tlist)
                except:
                    x = np.delete(x, [i,j])
                    y = np.delete(y, [i,j])
        m = ax.pcolor(x/1e6, y/1e6, Z, cmap=cm.coolwarm)
        ax.set_ylabel("$\Omega_c$ / MHz")
        ax.set_xlabel("$\Omega_p$ / MHz")
        plt.colorbar(m)
        plt.title(r"EIT transmission peak width against Rabi frequency")
        plt.show() 