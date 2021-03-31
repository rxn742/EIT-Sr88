#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:00:06 2021

@author: robgc
"""
import numpy as np
from scipy.constants import h, epsilon_0, c, hbar

def Isat(gamma, lam):
    return (np.pi*c*gamma*h)/(3*lam**3)

def rabsat(Ip, dme):
    return (dme/hbar)*np.sqrt((2*Ip)/(c*epsilon_0))