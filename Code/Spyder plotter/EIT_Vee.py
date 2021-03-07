#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:51:34 2020

@author: robgc
"""

import qutip as qt
qt.settings.auto_tidyup=False
import numpy as np
from scipy.constants import hbar, e, epsilon_0,c, m_e
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
from scipy.integrate import trapz
from scipy.signal import find_peaks, peak_widths
from shapely.geometry import LineString

"""defining the states"""
gstate = qt.basis(3,0) # ground state
rm = qt.basis(3,1) # red mot state
bm = qt.basis(3,2) # blue mot state

"""defining identities"""
gg = gstate*gstate.dag()
rmrm = rm*rm.dag()
bmbm = bm*bm.dag()

"""defining transition operators"""
grm = rm*gstate.dag() # ground to red mot
rmg = gstate*rm.dag() # red mot to ground
gbm = bm*gstate.dag() # ground to blue mot
bmg = gstate*bm.dag() # blue mot to ground

"""define experimental parameters"""

global drm
global dbm

pp = 1e-6 # Probe laser power in W
cp = 20e-3 # Coupling laser power in W
ld = 4e-3 # Laser beam diameter in m
Ip = pp/(np.pi*(ld/2)**2) # Probe intensity
Ic = cp/(np.pi*(ld/2)**2) # Coupling intensity
drm = 1.28e-30 # red mot dipole matrix element
dbm = np.sqrt((3*1.91*hbar*461e-9*e**2)/(4*np.pi*m_e*c)) # blue mot dipole matrix element
gammarm = 4.5e4 # Spontaneous emission rate from r to i
gammabm = 2*np.pi*32e6 # Spontaneous emission rate from i to g
Oc = (drm/hbar)*np.sqrt((2*Ic)/(c*epsilon_0*1.0003)) # Coupling Rabi frequency
Op = (dbm/hbar)*np.sqrt((2*Ip)/(c*epsilon_0*1.0003)) # Probe Rabi frequency
density = 1e15 # Number density of atoms in m^-3
lwc = 1e6 # Coupling laser linewidth in Hz
lwp = 1e6 # Probe laser linewidth in Hz
kp = 2*np.pi/461e-9 # Probe wavevector in m^-1
kc = 2*np.pi/689e-9 # Coupling wavevector in m^-1
delta_c = 0 # Coupling beam detuning in Hz

"""defining functions"""

def Hamiltonian(delta_p, delta_c, Omega_p, Omega_c):
    """
    This function defines the Hamiltonian of the 3 level system
    Parameters
    ----------
    delta_p : float
        Probe detuning in Hz.
    delta_c : float
        Coupling detuning in Hz.
    Omega_p : float
        Probe Rabi frequency in Hz.
    Omega_c : float
        Coupling Rabi frequency in Hz.

    Returns
    -------
    Qutip.Qobj (operator)
        Hamiltionian of the system (Uses Qutip convention hbar = 1)

    """
    return (-delta_p*(bmbm) - delta_c*(rmrm) + Omega_p*(gbm + bmg)/2 + Omega_c*(grm+rmg)/2)

def spon(gamma_rm, gamma_bm):
    """
    This function defines the spntaneous emission collapse operators
    Parameters
    ----------
    gamma_ri : float
        r-i spontaneous emission rate.
    gamma_ig : float
        i-g spontaneous emission rate.
    
    Returns
    -------
    list, dtype = Qutip.Qobj (operator)
        List of collapse operators for spontaneous emission

    """
    return [np.sqrt(gamma_rm)*rmg, np.sqrt(gamma_bm)*bmg]

def laser_linewidth(lwp, lwc):
    """
    Parameters
    ----------
    lwp : float
        Probe beam linewidth in Hz
    lwc : float
        Coupling beam linewidth in Hz

    Returns
    -------
    lw : numpy.ndarray, shape = 9x9, dtype = float64
        The laser linewidth super operator 

    """
    lw = np.zeros((9,9))
    lw[1,5] = -lwc
    lw[2,2] = -lwp
    lw[3,3] = -lwc
    lw[5,5] = -lwc-lwp
    lw[6,6] = -lwp
    lw[7,7] = -lwc-lwp
    return lw

def Liouvillian(delta_p, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc):
    """
    This function calculates the Liouvillian of the system 
    Parameters
    ----------
    delta_p : float
        Probe detuning in Hz.
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

    Returns
    -------
    L : Qutip.Qobj (super)
        The full Liouvillian super operator of the system for the master eqn

    """
    H = Hamiltonian(delta_p, delta_c, Omega_p, Omega_c)
    c_ops = spon(gamma_rm, gamma_bm)
    L = qt.liouvillian(H, c_ops)
    L_arr = L.data.toarray() # change to numpy array to add on laser linewidth matrix
    L_arr += laser_linewidth(lwp, lwc)
    L = qt.Qobj(L_arr, dims=[[[3], [3]], [[3], [3]]], type="super") # change back to Qobj
    return L

def population(delta_p, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc):
    """
    This function solves for the steady state density matrix of the system
    Parameters
    ----------
    delta_p : float
        Probe detuning in Hz.
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

    Returns
    -------
    rho : Qutip.Qobj (Density Matrix)
        The steady state density matrix of the 3 level system

    """
    rho = qt.steadystate(Liouvillian(delta_p, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc))
    return rho

def popgauss(delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, vlist):
    """
    This function generates the population probability for a given probe detuning with Doppler broadening
    Parameters
    ----------
elta_c : float
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
    vlist : numpy.ndarray, dtype = float
        A list of cross beam velocity groups 

    Returns
    -------
    pop : float
        State population probability for a given detuning

    """
    poplist = np.empty(len(vlist), dtype = complex) # list for chi values for each velocity group
    for i in range(len(vlist)):
        detuning = delta_p-kp*vlist[i]
        p = population(detuning, dclistP[i], Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)[state_index]
        poplist[i] = p
    i = poplist*normpdfP
    pop = trapz(i, vlist)
    return np.abs(pop)

def popcalc(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps):
    """
    This function generates an array of population values for a generated list of probe detunings
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
        Number of Probe detunings to calculate the population probability

    Returns
    -------
    dlist : numpy.ndarray, dtype = float64
        Array of Probe detunings
    plist : numpy.ndarray, dtype = float64
        Array of population probabilities corresponding to the detunings

    """
    global normpdfP # global variable for the gaussian distribution of velocity
    global dclistP # global variable for the list of Doppler shifted coupling detunings 
    
    plist = np.empty(steps+1)
    dlist = np.empty(steps+1)
    c = 0
    d=(dmax-dmin)*(steps)**(-1)
    gauss = input("Do you want to include a velocity distribution? \nY/N \n")
    if gauss == "Y":
        musig = input("Input mean and standard deviation transverse velocity \nmu, sig \n")
        musig = musig.split(",")
        mu = float(musig[0])
        sig = float(musig[1])
        vlist = np.linspace(-(mu+4*sig), mu+4*sig, 100) # list of possible velocity groups
        normpdfP = norm(mu, sig).pdf(vlist) # Gaussian distribution
        dclistP = np.empty(len(vlist))
        for i in range(len(vlist)):
            dclistP[i] = delta_c+kc*vlist[i]
        for i in range(0, steps+1):
            print(c)
            dlist[i] = dmin
            plist[i] = popgauss(dmin, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, vlist)
            dmin+=d
            c+=1
    else:
        for i in range(0, steps+1):
            dlist[i] = dmin
            plist[i] = population(dmin, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc)[state_index]
            dmin+=d
    return dlist, plist

def pop_plot(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin=-500e6, dmax=500e6, steps=2000):
    """
    This function plots the population probability of a chosen state against probe detuning
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
        Number of Probe detunings to calculate the population probabilities 

    Returns
    -------
    Population : plot
        Plot of chosen state population probability against probe detuning

    """
    global state_index
    state = input("Which state do you want to plot? \nGround, Red MOT, Blue MOT \n")
    if state == "Ground":
        state_index = 0,0
    if state == "Red MOT":
        state_index = 1,1
    if state == "Blue MOT":
        state_index = 2,2
    
    dlist, plist = popcalc(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(f"{state} population")
    ax.plot(dlist, plist, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$"\
            "\n" "$\Omega_p=$" f"{Omega_p/1e6:.1f} $MHz$" "\n" "$\Gamma_{rm}$" f"= {gammarm/(1e3*2*np.pi):.2f} $KHz$" "\n" "$\Gamma_{bm}$" f"= {gammabm/(1e6*2*np.pi):.2f} $MHz$" "\n" "$A_{rm}$" f" = {gammarm/1e4:.2f}"\
                r" x $10^4s^{-1}$" "\n""$A_{bm} =$"\
                    f"{gammabm/1e6:.1f}" r" x $10^6s^{-1}$" "\n" "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" f"$\gamma_p$ = {lwp/1e6:.1f} $MHz$" "\n"\
                        f"$\gamma_c$ = {lwc/1e6:.1f} $MHz$" "\n" f"Probe power = {pp*1e6:.1f} $\mu W$" "\n" f"Coupling power = {cp*1e3:.1f} $mW$")
    ax.set_xlabel(r"$\Delta_p$ / MHz")
    ax.set_ylabel(f"{state} state popultaion")
    ax.legend()
    plt.show()

def t(density, delta_p, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc):
    """
    This function calculates a transmission value for a given set of parameters
    Parameters
    ----------
    density : float
        Number density of atoms in the sample.   
    delta_p : float
        Probe detuning in Hz.
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

    Returns
    -------
    T : float
        Relative probe transmission value for the given parameters

    """
    p = population(delta_p, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc)[2,0] # element rho_ig
    chi = (-2*density*dbm**2*p)/(hbar*epsilon_0*Omega_p) # calculate susceptibility
    a = kp*np.abs(chi.imag) # absorption coefficient
    T = np.exp(-a*10e-3)
    return T

def tgauss(density, delta_p, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, vlist):
    """
    This function calculates a transmission value including Doppler broadening
    Parameters
    ----------
    density : float
        Number density of atoms in the sample.   
    delta_p : float
        Probe detuning in Hz.
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
    vlist : numpy.ndarray, dtype = float64
        An array of possible cross beam velocities

    Returns
    -------
    T : float
        Relative Doppler broadened probe transmission value for the given parameters  

    """
    chilist = np.empty(len(vlist), dtype = complex) # list for chi values for each velocity group
    for i in range(len(vlist)):
        detuning = delta_p-kp*vlist[i]
        p = population(detuning, dclistT[i], Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc)[2,0]
        chi = (-2*density*dbm**2*p)/(hbar*epsilon_0*Omega_p)
        chilist[i] = chi
    i = chilist*normpdfT
    chiavg = trapz(i, vlist)
    a = kp*np.abs(chiavg.imag)
    return np.exp(-a*3e-3)

def tcalcnod(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps):
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
        tlist[i] = t(density, dmin, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc)
        dmin+=d
    return dlist, tlist

def tcalc(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps):
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
    global normpdfT # global variable for the gaussian distribution of velocity
    global dclistT # global variable for the list of Doppler shifted coupling detunings 
    
    tlist = np.empty(steps+1)
    dlist = np.empty(steps+1)
    c = 0
    d=(dmax-dmin)*(steps)**(-1)
    gauss = input("Do you want to include a velocity distribution? \nY/N \n")
    if gauss == "Y":
        musig = input("Input mean and standard deviation transverse velocity \nmu, sig \n")
        musig = musig.split(",")
        mu = float(musig[0])
        sig = float(musig[1])
        vlist = np.linspace(-(mu+4*sig), mu+4*sig, 100) # list of possible velocity groups
        normpdfT = norm(mu, sig).pdf(vlist) # Gaussian distribution
        dclistT = np.empty(len(vlist))
        for i in range(len(vlist)):
            dclistT[i] = delta_c+kc*vlist[i]
        for i in range(0, steps+1):
            print(c)
            dlist[i] = dmin
            tlist[i] = tgauss(density, dmin, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, vlist)
            dmin+=d
            c+=1
    else:
        for i in range(0, steps+1):
            dlist[i] = dmin
            tlist[i] = t(density, dmin, delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc)
            dmin+=d
    return dlist, tlist
    
def trans_plot(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin=-500e6, dmax=500e6, steps=2000):
    """
    This function plots probe beam transmission for an array of probe detunings
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
    T : plot
        Plot of probe beam transmission against probe detuning, with EIT FWHM

    """
    dlist, tlist = tcalc(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps)
    """ Geometric library to calculate linewidth of EIT peak (FWHM) """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        pw = FWHM(dlist, tlist)[0]
        print(f"{pw:.2e}")
        ct = FWHM(dlist, tlist)[1]
        print(f"{ct:.2e}")
        ax.text(0.5, 0.4, f"EIT peak FWHM = {pw/(2*np.pi):.2f} $MHz$", transform=ax.transAxes, fontsize=10, va='center', ha='center')
        ax.text(0.5, 0.5, f"EIT peak contrast = {ct:.2e}", transform=ax.transAxes, fontsize=10, va='center', ha='center')    
    except:
        pass
    """Plotting"""
    plt.title(r"Probe transmission against probe beam detuning")
    ax.plot(dlist/(2e6*np.pi), tlist, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$"\
            "\n" "$\Omega_p=$" f"{Omega_p/1e3:.2f} $kHz$" "\n" "$\Gamma_{698}$" f"= {gamma_rm/(1e-3*2*np.pi):.2f} $mHz$" "\n" "$\Gamma_{461}$" f"= {gamma_bm/(1e6*2*np.pi):.2f} $MHz$"\
                    "\n" "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" f"$\gamma_p$ = {lwp/(2e-3*np.pi):.1f} $mHz$" "\n"\
                        f"$\gamma_c$ = {lwc/(2e-3*np.pi):.1f} $mHz$")
    ax.set_xlabel(r"$\Delta_p$/$2 \pi$ ($MHz$)")
    ax.set_ylabel(r"Probe Transmission")
    ax.legend()
    plt.show()
    
def reltrans_plot(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin=-500e6, dmax=500e6, steps=2000):
    """
    This function plots probe beam transmission for an array of probe detunings
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
    T : plot
        Plot of probe beam transmission against probe detuning, with EIT FWHM

    """
    dlist, tlist = tcalc(delta_c, Omega_p, Omega_c, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps)
    tlist2 = tcalc(delta_c, Omega_p, 0, gamma_rm, gamma_bm, lwp, lwc, dmin, dmax, steps)[1]
    """ Geometric library to calculate linewidth of EIT peak (FWHM) """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        pw, ct = FWHM(dlist, tlist)
        ax.text(f"EIT peak FWHM = {pw/(2*np.pi):.2f} $MHz$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    except:
        pass
    """Plotting"""
    plt.title(r"Probe transmission against probe beam detuning")
    ax.plot(dlist/(2e6*np.pi), tlist2, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$"\
            "\n" "$\Omega_p=$" f"{Omega_p/1e3:.2f} $kHz$" "\n" "$\Gamma_{rm}$" f"= {gamma_rm/(1e-3*2*np.pi):.2f} $mHz$" "\n" "$\Gamma_{bm}$" f"= {gamma_bm/(1e6*2*np.pi):.2f} $MHz$"\
                    "\n" "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" f"$\gamma_p$ = {lwp/(2e-3*np.pi):.1f} $mHz$" "\n"\
                        f"$\gamma_c$ = {lwc/(2e-3*np.pi):.1f} $mHz$")
    ax.set_xlabel(r"$\Delta_p$/$2 \pi$ ($MHz$)")
    ax.set_ylabel(r"Relative Probe Transmission")
    ax.legend()
    plt.show()    

def FWHM(dlist, tlist):
    """
    This function calculates the FWHM of the EIT peak in a spectrum
    Parameters
    ----------
    t : numpy.ndarray, dtype = float
        Calculated transmission values for a range of detunings

    Returns
    -------
    pw : float
        The FWHM of the EIT Peak in MHz

    """
    peak = find_peaks(tlist)[0]
    trough = find_peaks(-tlist)[0][0]
    ct = tlist[peak] - tlist[trough]
    width = peak_widths(tlist, peak)
    height = width[1]
    first_line = LineString(np.column_stack((dlist/1e6, np.full(len(tlist), height))))
    second_line = LineString(np.column_stack((dlist/1e6, tlist)))
    intersection = first_line.intersection(second_line)
    ints = []
    for i in intersection:
        ints.append(i.x)
    ints = np.array(ints)
    if len(ints) == 3:
        amax = np.argmax(np.abs(ints))
        ints = np.delete(ints, amax)
    if len(ints) == 4:
        amax = np.argmax(np.abs(ints))
        ints = np.delete(ints, amax)
        amax = np.argmax(np.abs(ints))
        ints = np.delete(ints, amax)
    i1 = ints[0]
    i2 = ints[1]
    if i1 == 0.0:
        pw = np.abs(i2)
    if i1 == 0.0:
        pw = np.abs(i2)
    if i1 < 0 :
        pw = np.abs(i1-i2)
    if i1 > 0:
        pw = i2-i1
    return [pw, ct[0]]

def optimize1D(delta_c, gamma_rm, gamma_bm, lwp, lwc, points=1000):
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
            pop = np.abs(population(-delta_c, delta_c, Omega_p[i], Oc, gamma_rm, gamma_bm, lwp, lwc)[2,2])
            rp.append(pop)
    if rabi == "Coupling":
        Omega_c = np.linspace(float(ran[0]), float(ran[1]), points)
        Omega_p = Op
        for i in range(len(Omega_c)):
            pop = np.abs(population(-delta_c, delta_c, Op, Omega_c[i], gamma_rm, gamma_bm, lwp, lwc)[2,2])
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
    
def optimize2D(delta_c, gamma_rm, gamma_bm, lwp, lwc, diameter, points=100):
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
    Omega_c = (drm/hbar)*np.sqrt((2*Ic)/(c*epsilon_0*1.0003)) # Coupling Rabi frequency
    Omega_p = (dbm/hbar)*np.sqrt((2*Ip)/(c*epsilon_0*1.0003)) # Probe Rabi frequency
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    x, y = np.meshgrid(Omega_p, Omega_c)
    Z = np.zeros((len(y), len(x)))
    if opt == "Rydberg":
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i,j] = np.abs(population(-delta_c, delta_c, x[i,j], y[i,j], gamma_rm, gamma_bm, lwp, lwc)[2,2])
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
                Z[i,j] = np.abs(t(density, -delta_c, delta_c, x[i,j], y[i,j], gamma_rm, gamma_bm, lwp, lwc))
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
                dlist, tlist =  tcalcnod(delta_c, x[i,j], y[i,j], gamma_rm, gamma_bm, lwp, lwc, -50e6, 50e6, 100)
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