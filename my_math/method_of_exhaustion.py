#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method of exhaustion for determining a function's integral.

@author: Christian Ruf
"""
import numpy as np


def square(x):
    return x*x


def cube(x):
    return x**3


def get_area(fkt, a, b, eps=0.01):
    intervall_size = (b-a) / 1
    ober_summe = 100000
    unter_summe = 0 
    while ober_summe - unter_summe > eps:
        intervall_size /= 100
        xs = np.arange(a, b, intervall_size)
        unter_summe = (fkt(xs) * intervall_size).sum()
        ober_summe = (fkt(xs + intervall_size) * intervall_size).sum()
    return (ober_summe + unter_summe) / 2
    
get_area(square, 0, 3)  # 9
get_area(cube, 0, 3)  # ~20
get_area(cube, -3, 3)  # 0
