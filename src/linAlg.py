#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lineare Algebra
"""
import numpy as np
from sympy import Matrix


A = np.array([[2**n for n in range(5)],
              [5**n for n in range(5)],
              [6**n for n in range(5)]]) 
A = Matrix(A)
b = np.zeros((3,1))

np.array([2**n for n in range(5)]) - np.array([5**n for n in range(5)])
np.array([2**n for n in range(5)]) - np.array([6**n for n in range(5)])



a0 = 1
a3 = 14
a4 = -2
a1 = 52*a3 +616*a4
a2 = -13*a3 -117*a4


def p(z):
    return a0 + a1*z + a2*z**2 + a3*z**3 + a4*z**4

p(2)
p(5)
p(6)
