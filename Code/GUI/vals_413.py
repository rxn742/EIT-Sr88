#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:15:45 2021

@author: robgc
"""
import numpy as np
from scipy.constants import hbar, e, epsilon_0,c, m_e

""" 413nm Strontium """
d23_413 = np.sqrt((3*1.6e-4*hbar*413.3e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
d12_413 = 4.469788031e-29 # i-g dipole matrix element
spontaneous_32_413 = 4e4 # Spontaneous emission rate from r to i
spontaneous_21_413 = 2*np.pi*32e6*0.99998 # Spontaneous emission rate from i to g
kp_413 = 2*np.pi/461e-9 # Probe wavevector in m^-1
kc_413 = 2*np.pi/413e-9 # Coupling wavevector in m^-1

def func_omega_c_413(Ic):
    return (d23_413/hbar)*np.sqrt((2*Ic)/(c*epsilon_0))

def func_omega_p_413(Ip):
    return (d12_413/hbar)*np.sqrt((2*Ip)/(c*epsilon_0))

def func_Ic_413(cp, cd):
    return cp/(np.pi*(cd/2)**2)

def func_Ip_413(pp, pd):
    return pp/(np.pi*(pd/2)**2)
