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
import matplotlib.ticker as mticker
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import itertools
from scipy.integrate import trapz
from scipy.signal import find_peaks, peak_widths
from shapely.geometry import LineString

"""defining the states"""
gstate = qt.basis(3,0) # ground state
istate = qt.basis(3,1) # intermediate state
rstate = qt.basis(3,2) # excited state

"""defining identities"""
gg = gstate*gstate.dag()
ii = istate*istate.dag()
rr = rstate*rstate.dag()

"""defining ladder operators"""
gi = istate*gstate.dag() # ground to intermediate
ig = gstate*istate.dag() # intermediate to ground
ir = rstate*istate.dag() # intermediate to excited
ri = istate*rstate.dag() # excited to intermediate

pp = 1e-6
cp = 250e-3
d = 4e-3
Ip = pp/(np.pi*(d/2)**2)
Ic = cp/(np.pi*(d/2)**2)
dri = np.sqrt((3*1.6e-4*hbar*413.3e-9*e**2)/(4*np.pi*m_e*c))
dig = np.sqrt((3*1.91*hbar*461e-9*e**2)/(4*np.pi*m_e*c))
gri = 4e4
gig = 2*np.pi*32e6
Oc = (dri/hbar)*np.sqrt((2*Ic)/(c*epsilon_0*1.0003))
Op = (dig/hbar)*np.sqrt((2*Ip)/(c*epsilon_0*1.0003))
density = 1e15
lwc = 1e6
lwp = 1e6
kp = 2*np.pi/461e-9
kc = 2*np.pi/413e-9


"""defining hamiltonian"""

def Hamiltonian(delta_p, delta_c, Omega_p, Omega_c):
    return (-delta_p*(ii + rr) - delta_c*(rr) + Omega_p*(gi + ig)/2 + Omega_c*(ir+ri)/2)

"""defining collapse operators"""
def J_ri(gamma_ri):
    return np.sqrt(gamma_ri)*ri
def J_ig(gamma_ig):
    return np.sqrt(gamma_ig)*ig

def laser_linewidth(lwp, lwc):
    lw = np.zeros((9,9))
    lw[1,5] = -lwp
    lw[2,2] = -lwp-lwc
    lw[3,3] = -lwp
    lw[5,5] = -lwc
    lw[6,6] = -lwp-lwc
    lw[7,7] = -lwc
    return lw

def Liouvillian(delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc):
    H = Hamiltonian(delta_p, delta_c, Omega_p, Omega_c)
    c_ops = [J_ri(gamma_ri), J_ig(gamma_ig)]
    L = qt.liouvillian(H, c_ops)
    L_arr = L.data.toarray()
    L_arr = L_arr +laser_linewidth(lwp, lwc)
    L = qt.Qobj(L_arr, dims=[[[3], [3]], [[3], [3]]], type="super")
    return L

"""steady state solution"""
def population(delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc):  
    L = qt.steadystate(Liouvillian(delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc))
    return L

def t(density, delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc):
    p = population(delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)[1,0]
    chi = (-2*density*dig**2*p)/(hbar*epsilon_0*Omega_p)
    a = kp*np.abs(chi.imag)
    return np.exp(-a*3e-3)

def tgauss(density, delta_p, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, x):
    chilist = np.empty(len(x), dtype = complex)
    for i in range(len(x)):
        detuning = delta_p-kp*x[i]
        p = population(detuning, dclist[i], Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)[1,0]
        chi = (-2*density*dig**2*p)/(hbar*epsilon_0*Omega_p)
        chilist[i] = chi
    i = chilist*normpdf
    chiavg = trapz(i, x)
    a = kp*np.abs(chiavg.imag)
    return np.exp(-a*3e-3)

def tcalc(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, dmin=-600e6, dmax=600e6, steps=1000):
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
        x = np.linspace(-(mu+4*sig), mu+4*sig, 100)
        global normpdf
        normpdf = norm(mu, sig).pdf(x)
        global dclist
        dclist = np.empty(len(x))
        for i in range(len(x)):
            dclist[i] = delta_c+kc*x[i]
        for i in range(0, steps+1):
            print(c)
            dlist[i] = dmin
            tlist[i] = tgauss(density, dmin, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, x)
            dmin+=d
            c+=1
    else:
        for i in range(0, steps+1):
            dlist[i] = dmin
            tlist[i] = t(density, dmin, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)
            dmin+=d
    return dlist, tlist

def pop_plot(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, dmin=-600e6, dmax=600e6, steps=1000):
    state = input("Which state do you want to plot? \nGround, Intermediate, Rydberg \n")
    if state == "Ground":
        k = 0,0
    if state == "Intermediate":
        k = 1,1
    if state == "Rydberg":
        k = 2,2
    d=(dmax-dmin)*(steps)**(-1)
    ilist = []
    dlist = []
    for i in range(0, steps+1):
        dlist.append(dmin)
        ilist.append(population(dmin, delta_c, Omega_p,Omega_c,gamma_ri,gamma_ig, lwp, lwc)[k])
        dmin+=d
    
    dlist = np.array(dlist)
    ilist = np.array(ilist)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(f"{state} population")
    ax.plot(dlist/gig, ilist, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$" "\n" \
            "$\Omega_p=$" f"{Omega_p/1e6:.1f} $MHz$" "\n" "$\Gamma_{ri}$" f"= {gri/1e6:.2f} $MHz$" "\n"\
                "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" "$\Gamma_{ig} =$" f"{gig/1e6:.1f} $MHz$" "\n"\
                    f"$\gamma_p$ = {lwp/1e6:.1f} MHz" "\n" f"$\gamma_c$ = {lwc/1e6:.1f} MHz")
    ax.set_xlabel(r"$\Delta_p/\Gamma_{ig}$")
    ax.set_ylabel(f"{state} state popultaion")
    ax.legend()
    plt.show()
    
def trans_plot(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, dmin=-600e6, dmax=600e6, steps=1000):
    dlist, tlist = tcalc(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, dmin, dmax, steps)
    F = FWHM(tlist)
    first_line = LineString(np.column_stack((dlist/1e6, np.full(len(tlist), F[0]))))
    second_line = LineString(np.column_stack((dlist/1e6, tlist)))
    intersection = first_line.intersection(second_line)
    peak_width = ((np.abs(intersection[1].x) - np.abs(intersection[2].x)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(r"Probe transmission against probe beam detuning")
    ax.plot(dlist/1e6, tlist, color="orange", label="$\Omega_c=$" f"{Omega_c/1e6:.1f} $MHz$"\
            "\n" "$\Omega_p=$" f"{Omega_p/1e6:.1f} $MHz$" "\n" "$\Gamma_{ri}$" f"= {gri/(1e3*2*np.pi):.2f} $KHz$" "\n" "$\Gamma_{ig}$" f"= {gig/(1e6*2*np.pi):.2f} $MHz$" "\n" "$A_{r}$" f" = {gri/1e4:.2f}"\
                r" x $10^4s^{-1}$" "\n""$A_{i} =$"\
                    f"{gig/1e6:.1f}" r" x $10^6s^{-1}$" "\n" "$\Delta_c =$" f"{delta_c/1e6:.1f} $MHz$" "\n" f"$\gamma_p$ = {lwp/1e6:.1f} $MHz$" "\n"\
                        f"$\gamma_c$ = {lwc/1e6:.1f} $MHz$" "\n" f"Probe power = {pp*1e6:.1f} $\mu W$" "\n" f"Coupling power = {cp*1e3:.1f} $mW$")
    ax.set_xlabel(r"$\Delta_p$ / $MHz$")
    ax.set_ylabel(r"Probe Transmission")
    ax.text(0.4, 0.65, f"EIT peak FWHM = {peak_width:.2f} $MHz$", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.legend()
    plt.show()
    
def trans_peak(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc):
    peak = t(density, -delta_c, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)
    return peak

def rydberg_peak(delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc):
    peak = np.abs(population(-delta_c, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc)[2,2])
    return peak

def optimize1D(delta_c, gamma_ri, gamma_ig, lwp, lwc, points=1000):
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
    
def optimize2D(delta_c, gamma_ri, gamma_ig, lwp, lwc, points=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Omega_c = np.linspace(1e6, 1e8, points)
    Omega_p = np.linspace(1e6, 1e10, points)
    q = list(itertools.product(Omega_p, Omega_c))
    x, y = np.meshgrid(Omega_p, Omega_c)
    z = []
    for i in range(len(q)):
        pop = np.abs(population(-delta_c, delta_c, q[i][0], q[i][1], gamma_ri, gamma_ig, lwp, lwc)[2,2])
        z.append(pop)
    Z = np.array(z)
    Z = Z.reshape((len(x), len(y)))
    ax.plot_surface(x/1e6, y/1e6, Z, cmap=cm.coolwarm)
    ax.set_ylabel("$\Omega_c$ / MHz")
    ax.set_xlabel("$\Omega_p$ / MHz")
    ax.set_zlabel("Rydberg state population")
    plt.show()
    
def FWHM(t):
    peak = find_peaks(t)[0]
    width = peak_widths(t, peak)
    height = width[1]
    return height