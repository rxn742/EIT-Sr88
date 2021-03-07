#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:58:40 2021

@author: robgc
"""
import numpy as np
import matplotlib.pyplot as plt
from EIT_Ladder2 import maxwell_long, maxwell_trans, v_mp

vlong = np.linspace(0, 1200, 5000)
v_trans = np.linspace(-65, 65, 5000)
v_mp_trans = np.sqrt(3/2)*50e-3*v_mp(623.15)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(vlong, maxwell_long(vlong, v_mp(623.15)), label="Oven at $350 ^o C$")
ax1.set_xlabel("Longitudinal Velocity (m/s)")
ax1.set_ylabel("Probability Density")
plt.title("Longitudinal Velocity Distribution")
plt.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(v_trans, maxwell_trans(v_trans, v_mp_trans), label="Oven at $350 ^o C$")
ax2.set_xlabel("Transverse Velocity (m/s)")
ax2.set_ylabel("Probability Density")
plt.legend()
plt.title("Transverse Velocity Distribution")