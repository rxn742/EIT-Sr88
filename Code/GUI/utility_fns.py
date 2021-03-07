#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:00:06 2021

@author: robgc
"""
import numpy as np
from scipy.constants import hbar, c

def Isat(gamma, lam):
    return (2*np.pi**2*c*gamma*hbar)/(3*lam**3)