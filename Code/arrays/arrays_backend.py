#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:26:13 2021

@author: robgc
"""
import sys
sys.path.insert(0,'../GUI')
import numpy as np
from backend import tcalc, pop_calc
from vals_413 import d12_413, d23_413, spontaneous_21_413, spontaneous_32_413, kp_413, kc_413, \
                        func_Ic_413, func_Ip_413, func_omega_c_413, func_omega_p_413
import multiprocessing as mp

diameters = np.linspace(0.1e-3, 7e-3, 9)

    
def my_func(i):
    d, t = tcalc(0, 10e6, 30e6, spontaneous_32_413, spontaneous_21_413, 1e5, 
                 1e5, -157e6, 157e6, 2000, "N", kp_413, kc_413, 1e15, 
                 d12_413, 3e-3, 623.15, 50e-3, i, 5e-3, "Y")
    #d, p = pop_calc(0, 10e6, 30e6, spontaneous_32_413, spontaneous_21_413, 1e5, 
    #             1e5, -157e6, 157e6, 2000, (2,2), "N", 623.15, kp_413, kc_413,
    #             50e-3, 1e-3, i, "Y")
    np.savetxt(f'pdiam_t_{i/1e-3:.1f}mm.csv', t, delimiter=',')
    #np.savetxt(f'pdiam_p_{i/1e-3:.1f}mm.csv', p, delimiter=',')
    print(f"done {i/1e-3:.1f}")

def main():
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_func, diameters)
    print("done all")