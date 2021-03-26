#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:15:45 2021

@author: robgc
"""
import numpy as np
from scipy.constants import hbar, e, epsilon_0,c, m_e
""" Unhash to plot graphs"""
#from EIT_Ladder import * 

""" Hash out below all but one set of variables for chosen transitions"""

""" 413nm Strontium """
dri = np.sqrt((3*1.6e-4*hbar*413.3e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
dig = np.sqrt((3*1.91*hbar*461e-9*e**2)/(4*np.pi*m_e*c)) # i-g dipole matrix element
gamma_ri = 4e4 # Spontaneous emission rate from r to i
gamma_ig = 2*np.pi*32e6*0.99998 # Spontaneous emission rate from i to g
kp = 2*np.pi/461e-9 # Probe wavevector in m^-1
kc = 2*np.pi/413e-9 # Coupling wavevector in m^-1
pp = 10e-6 # Probe laser power in W
cp = 50e-3 # Coupling laser power in W
cd = 1e-3 # Laser beam diameter in m
pd = 4e-3
Ip = pp/(np.pi*(pd/2)**2) # Probe intensity
Ic = cp/(np.pi*(cd/2)**2) # Coupling intensity
Omega_c = (dri/hbar)*np.sqrt((2*Ic)/(c*epsilon_0)) # Coupling Rabi frequency
Omega_p = (dig/hbar)*np.sqrt((2*Ip)/(c*epsilon_0)) # Probe Rabi frequency
density = 1e15 # Number density of atoms in m^-3
lwc = 1e6 # Coupling laser linewidth in Hz
lwp = 1e6 # Probe laser linewidth in Hz
sl = 10e-3 # Sample length of atoms
beamdiv = 38e-3 # Divergence angle of the atomic beam

""" 318nm Strontium """
#dri = np.sqrt((3*2.6e-6*hbar*318.5e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
#dig = np.sqrt((3*1.91*hbar*461e-9*e**2)/(4*np.pi*m_e*c)) # i-g dipole matrix element
#gamma_ri = 4e4 # Spontaneous emission rate from r to i
#gamma_ig = 4.69e4 # Spontaneous emission rate from i to g
#kp = 2*np.pi/689e-9 # Probe wavevector in m^-1
#kc = 2*np.pi/318e-9 # Coupling wavevector in m^-1

""" Mauger et al """
#dri = np.sqrt((3*1.6e-4*hbar*413.3e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
#dig = np.sqrt((3*1.91*hbar*461e-9*e**2)/(4*np.pi*m_e*c)) # i-g dipole matrix element
#gamma_ri = 2*np.pi*3.5e6 # Spontaneous emission rate from r to i
#gamma_ig = 2*np.pi*16e6*0.99995 # Spontaneous emission rate from i to g
#kp = 2*np.pi/461e-9 # Probe wavevector in m^-1
#kc = 2*np.pi/413e-9 # Coupling wavevector in m^-1
#Omega_c = 2*np.pi*7.5e6 # Coupling Rabi frequency
#Omega_p = 2*np.pi*15.1e6 # Probe Rabi frequency
#density = 1e15 # Number density of atoms in m^-3
#lwc = 1e6 # Coupling laser linewidth in Hz
#lwp = 1e6 # Probe laser linewidth in Hz
#sl = 3e-3 # Sample length of atoms

""" Mauger et al """
#dri = np.sqrt((3*1.6e-4*hbar*413.3e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
#dig = np.sqrt((3*1.91*hbar*461e-9*e**2)/(4*np.pi*m_e*c)) # i-g dipole matrix element
#gamma_ri = 2*np.pi*14e3 # Spontaneous emission rate from r to i
#gamma_ig = 2*np.pi*32e6 # Spontaneous emission rate from i to g
#kp = 2*np.pi/461e-9 # Probe wavevector in m^-1
#kc = 2*np.pi/413e-9 # Coupling wavevector in m^-1
#Omega_c = 2*np.pi*3.1e6 # Coupling Rabi frequency
#Omega_p = 2*np.pi*13.1e6 # Probe Rabi frequency
#density = 1.7e14 # Number density of atoms in m^-3
#lwc = 2*np.pi*0.6e6 # Coupling laser linewidth in Hz
#lwp = 2*np.pi*0.7e6 # Probe laser linewidth in Hz
#sl = 30e-3 # Sample length of atoms

""" For population plot run pop_plot(delta_c, dmin, dmax, steps)"""
""" For transmission plot run trans_plot(delta_c, dmin, dmax, steps)"""