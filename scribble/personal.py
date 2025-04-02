#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:01:05 2019
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# binomial
n = 15
p = 0.1
x = np.arange(stats.binom.ppf(0.01, n, p), stats.binom.ppf(0.99, n, p))

fig, ax = plt.subplots(1, 1)
ax.plot(x, stats.binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')

# Annäherung durch Normalverteilung
mu = n * p
variance = mu * (1 - p)
sigma = math.sqrt(variance)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

# Annäherung durch Poisson (auch diskret, aber lambda reell)
lambd = n * p
x = np.arange(stats.poisson.ppf(0.01, lambd), stats.poisson.ppf(0.99, lambd))
plt.plot(x, stats.poisson.pmf(x, lambd), 'ro', ms=8, label='poisson pmf')
plt.show()
