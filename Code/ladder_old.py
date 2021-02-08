#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:19:18 2021

@author: robgc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:51:34 2020
@author: robgc
"""

import qutip as qt
qt.settings.auto_tidyup=False
import numpy as np
from scipy.constants import hbar, epsilon_0
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapz
from scipy.signal import find_peaks, peak_widths
from shapely.geometry import LineString
from plotter import dig, gamma_ri, gamma_ig, pp, cp, Omega_c, Omega_p, density, lwp, lwc, kp, kc

"""defining the states"""
gstate = qt.basis(3,0) # ground state
istate = qt.basis(3,1) # intermediate state
rstate = qt.basis(3,2) # excited state

"""defining identities"""
gg = gstate*gstate.dag()
ii = istate*istate.dag()
rr = rstate*rstate.dag()

"""defining transition operators"""
gi = istate*gstate.dag() # ground to intermediate
ig = gstate*istate.dag() # intermediate to ground
ir = rstate*istate.dag() # intermediate to excited
ri = istate*rstate.dag() # excited to intermediate

"""defining functions"""

def Hamiltonian(delta_p, delta_c):
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
    return (-delta_p*(ii + rr) - delta_c*(rr) + Omega_p*(gi + ig)/2 + Omega_c*(ir+ri)/2)

def spon():
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
    return [np.sqrt(gamma_ri)*ri, np.sqrt(gamma_ig)*ig]

def laser_linewidth():
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
    lw[1,5] = -lwp
    lw[2,2] = -lwp-lwc
    lw[3,3] = -lwp
    lw[5,5] = -lwc
    lw[6,6] = -lwp-lwc
    lw[7,7] = -lwc
    return lw

def Liouvillian(delta_p, delta_c):
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
    H = Hamiltonian(delta_p, delta_c)
    c_ops = spon()
    L = qt.liouvillian(H, c_ops)
    L_arr = L.data.toarray() # change to numpy array to add on laser linewidth matrix
    L_arr += laser_linewidth()
    L = qt.Qobj(L_arr, dims=[[[3], [3]], [[3], [3]]], type="super") # change back to Qobj
    return L

def population(delta_p, delta_c):
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
    rho = qt.steadystate(Liouvillian(delta_p, delta_c))
    return rho

def popgauss(delta_p, delta_c, vlist):
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
        p = population(detuning, dclistP[i])[state_index]
        poplist[i] = p
    i = poplist*normpdfP
    pop = trapz(i, vlist)
    return np.abs(pop)

def popcalc(delta_c, dmin, dmax, steps):
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
    count = 0
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
            print(count)
            dlist[i] = dmin
            plist[i] = popgauss(dmin, delta_c, vlist)
            dmin+=d
            count+=1
    else:
        for i in range(0, steps+1):
            dlist[i] = dmin
            plist[i] = population(dmin, delta_c)[state_index]
            dmin+=d
    return dlist, plist

def pop_plot(delta_c, dmin=-500e6, dmax=500e6, steps=2000):
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
    state = input("Which state do you want to plot? \nGround, Intermediate, Rydberg \n")
    if state == "Ground":
        state_index = 0,0
    if state == "Intermediate":
        state_index = 1,1
    if state == "Rydberg":
        state_index = 2,2
    
    dlist, plist = popcalc(delta_c, dmin, dmax, steps)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(f"{state} population")
    ax.plot(dlist, plist, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$"\
            "\n" "$\Omega_p=$" f"{Omega_p/1e6:.1f} $MHz$" "\n" "$\Gamma_{ri}$" f"= {gamma_ri/(1e3*2*np.pi):.2f} $KHz$" "\n" "$\Gamma_{ig}$" f"= {gamma_ig/(1e6*2*np.pi):.2f} $MHz$" "\n" "$A_{r}$" f" = {gamma_ri/1e4:.2f}"\
                r" x $10^4s^{-1}$" "\n""$A_{i} =$"\
                    f"{gamma_ig/1e6:.1f}" r" x $10^6s^{-1}$" "\n" "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" f"$\gamma_p$ = {lwp/1e6:.1f} $MHz$" "\n"\
                        f"$\gamma_c$ = {lwc/1e6:.1f} $MHz$" "\n" f"Probe power = {pp*1e6:.1f} $\mu W$" "\n" f"Coupling power = {cp*1e3:.1f} $mW$")
    ax.set_xlabel(r"$\Delta_p$ / MHz")
    ax.set_ylabel(f"{state} state popultaion")
    ax.legend()
    plt.show()

def transmission(delta_p, delta_c):
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
    p = population(delta_p, delta_c)[1,0] # element rho_ig
    chi = (-2*density*dig**2*p)/(hbar*epsilon_0*Omega_p) # calculate susceptibility
    a = kp*np.abs(chi.imag) # absorption coefficient
    T = np.exp(-a*3e-3)
    return T

def tgauss(delta_p, delta_c, vlist):
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
        p = population(detuning, dclistT[i])[1,0]
        chi = (-2*density*dig**2*p)/(hbar*epsilon_0*Omega_p)
        chilist[i] = chi
    i = chilist*normpdfT
    chiavg = trapz(i, vlist)
    a = kp*np.abs(chiavg.imag)
    return np.exp(-a*3e-3)

def tcalc(delta_c, dmin, dmax, steps):
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
    count = 0
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
            print(count)
            dlist[i] = dmin
            tlist[i] = tgauss(dmin, delta_c, vlist)
            dmin+=d
            count+=1
    else:
        for i in range(0, steps+1):
            dlist[i] = dmin
            tlist[i] = transmission(dmin, delta_c)
            dmin+=d
    return dlist, tlist
    
def trans_plot(delta_c, dmin=-500e6, dmax=500e6, steps=2000):
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
    dlist, tlist = tcalc(delta_c, dmin, dmax, steps)
    """ Plotting"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    """ Geometric library to calculate linewidth of EIT peak (FWHM) """
    try:
        pw = FWHM(dlist, tlist)
        ax.text(0.5, 0.97, f"EIT peak FWHM = {pw:.2f} $MHz$", transform=ax.transAxes, fontsize=10, va='center', ha='center')
    except:
        pass
    
    plt.title(r"Probe transmission against probe beam detuning")
    ax.plot(dlist/1e6, tlist, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$"\
            "\n" "$\Omega_p=$" f"{Omega_p/1e6:.1f} $MHz$" "\n" "$\Gamma_{ri}$" f"= {gamma_ri/(1e3*2*np.pi):.2f} $KHz$" "\n" "$\Gamma_{ig}$" f"= {gamma_ig/(1e6*2*np.pi):.2f} $MHz$" "\n" "$A_{r}$" f" = {gamma_ri/1e4:.2f}"\
                r" x $10^4s^{-1}$" "\n""$A_{i} =$"\
                    f"{gamma_ig/1e6:.1f}" r" x $10^6s^{-1}$" "\n" "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" f"$\gamma_p$ = {lwp/1e6:.1f} $MHz$" "\n"\
                        f"$\gamma_c$ = {lwc/1e6:.1f} $MHz$" "\n" f"Probe power = {pp*1e6:.1f} $\mu W$" "\n" f"Coupling power = {cp*1e3:.1f} $mW$")
    ax.set_xlabel(r"$\Delta_p$ / $MHz$")
    ax.set_ylabel(r"Probe Transmission")
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
    return pw