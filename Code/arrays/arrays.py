#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:26:13 2021

@author: robgc
"""
import sys
sys.path.insert(0,'..')
import numpy as np
from EIT_Ladder_noinput import tcalc
import multiprocessing as mp
from plotter import Omega_p, Omega_c, density

#mslist = np.asarray([[0,0.1],[0,0.2],[0,0.3],[0,0.4],[0,0.5],[0,0.6],[0,0.7],[0,0.8],[0,0.9]])
#for i in mslist:
#    d1, t1 = tcalc(0, -200e6, 200e6, 2000, i, "co")
#    np.savetxt(f'co_background{i[1]}.csv', t1, delimiter=',')
#    d2, t2 = tcalc(0, -200e6, 200e6, 2000, i, "count")
#    np.savetxt(f'count_{i[1]}.csv', t1, delimiter=',')

rfs = np.linspace(1e-3, 20e-3, 20)


#for i in mslist:
#    d2, t2 = tcalc(0, -200e6, 200e6, 2000, i)
#    np.savetxt(f'count_back_{i[1]}.csv', t2, delimiter=',')
    
def my_func(i):
    sl = i
    d, t = tcalc(0, Omega_p, 0, -200e6, 200e6, 2000, density, sl)
    np.savetxt(f'beamwidth_back{i/1e-3:.1f}mm.csv', t, delimiter=',')
    print(f"done {i/1e-3:.1f}")
    
def main():
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_func, rfs)
    print("done all")
    