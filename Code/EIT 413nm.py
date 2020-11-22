#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 15:51:34 2020

@author: robgc
"""

import qutip as qt
qt.settings.auto_tidyup=False
import numpy as np
from scipy.constants import hbar, e, epsilon_0
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

gri = 4e4
gig = 2*np.pi*32e6
Oc = 7e6
Op = 7e6/5
dig=5.249*8.4783536255e-30
density = 1e15

"""defining hamiltonian"""

def Hamiltonian(delta_p, delta_c, Omega_p, Omega_c):
    return (-delta_p*(ii + rr) - delta_c*(rr) + Omega_p*(gi + ig) + Omega_c*(ir+ri))

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
    a = 2*np.pi*np.abs(chi.imag)/461e-9
    #print(a)
    return np.exp(-a*1e-4)

def plot(state, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc, dmin=-2, dmax=2, steps=1000):
    if state == "ground":
        k = 0,0
    if state == "intermediate":
        k = 1,1
    if state == "excited":
        k = 2,2
    d=(dmax-dmin)*(steps)**(-1)
    ilist = []
    dlist = []
    tlist = []
    for i in range(0, steps+1):
        dlist.append(dmin)
        #ilist.append(population(dmin, delta_c, Omega_p,Omega_c,gamma_ri,gamma_ig)[k])
        tlist.append(t(density, dmin, delta_c, Omega_p, Omega_c, gamma_ri, gamma_ig, lwp, lwc))
        dmin+=d
    
    dlist = np.array(dlist)
    ilist = np.array(ilist)
    tlst = np.array(tlist)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(r"Transmission with $\Omega_c = 7MHz$")
    ax.plot(dlist/gig, tlist, color="orange", label="$\Omega_p= 1.4MHz$ \n" "$\Gamma_{ri}$" f"= {gri/1e6:.3} $MHz$" "\n" "$\Delta_c =$" f"{delta_c/1e6:.3} $MHz$" "\n" "$\Gamma_{ig} =$" f"{200} $MHz$" "\n" "Probe linewidth = 1MHz" "\n" "Coupling linewidth = 1MHz")
    ax.set_xlabel(r"$\Delta_p/\Gamma_{ig}$")
    ax.set_ylabel(r"Probe Transmission")
    ax.legend()
    plt.show()
    
    