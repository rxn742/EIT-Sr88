#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:31:23 2020

@author: robgc
"""
import numpy as np
import os
from pairinteraction import pireal as pi
from scipy.constants import hbar

if not os.path.exists("./cache"):
    os.makedirs("./cache")
cache = pi.MatrixElementCache("./cache")

state_g = pi.StateOne("Sr1", 3, 0, 0, 0)
state_i = pi.StateOne("Sr1", 5, 1, 1, 0)
state_r = pi.StateOne("Sr1", 56, 2, 2, 0)

dig = cache.getElectricDipole(state_i, state_g)*hbar*1e7
dri = cache.getElectricDipole(state_r, state_i)*hbar*1e7