#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 23:15:45 2021

@author: robgc
"""
import numpy as np
from scipy.constants import hbar, e, epsilon_0,c, m_e

""" 413nm Strontium """
d23_318 = np.sqrt((3*2.6e-6*hbar*318e-9*e**2)/(4*np.pi*m_e*c)) # r-i dipole matrix element
d12_318 = 1.339579873e-30 # i-g dipole matrix element
spontaneous_32_318 = 5.1e4 # Spontaneous emission rate from r to i
spontaneous_21_318 = 4.69e4 # Spontaneous emission rate from i to g
kp_318 = 2*np.pi/689e-9 # Probe wavevector in m^-1
kc_318 = 2*np.pi/318e-9 # Coupling wavevector in m^-1

def func_omega_c_318(Ic):
    return (d23_318/hbar)*np.sqrt((2*Ic)/(c*epsilon_0))

def func_omega_p_318(Ip):
    return (d12_318/hbar)*np.sqrt((2*Ip)/(c*epsilon_0))

def func_Ic_318(cp, cd):
    return cp/(np.pi*(cd/2)**2)

def func_Ip_318(pp, pd):
    return pp/(np.pi*(pd/2)**2)
