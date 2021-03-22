#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:04:53 2021

@author: robgc
"""
import cProfile
import pstats
from backend import tcalc
import numpy as np

def profile1():
    pr = cProfile.Profile()
    pr.enable()
    tcalc(0, 10e6, 25e6, 4e4, 201e6, 1e6, 1e6, -50*2e6*np.pi, 50*2e6*np.pi, 1000, "Y", 2*np.pi/461e-9, 2*np.pi/461e-9, 1e15, 4.47e-29, 3e-3, 623.15, 50e-3, 4e-3, 1e-3, "N")
    pr.disable()
    return pstats.Stats(pr)

if __name__ == "__main__":
    profile1().sort_stats('time').print_stats()