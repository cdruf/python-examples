#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:06:24 2020

@author: Christian
"""

import numpy as np
import matplotlib.pyplot as plt


#%%
# Simpel
f = lambda x: x**2
fx = lambda x: 2*x
alpha = 0.1
theta = 100
path = np.array([])
for i in range(1000):
    theta = theta - alpha * fx(theta)
    path = np.append(path, theta)
    
x = np.arange(-20, 100, 0.01)    
plt.plot(x, f(x), path, f(path), '.')


#%% 
# Simpel 2
f = lambda x,y: (x+y)**2
fx = lambda x,y: 2*(x+y)
fy = lambda x,y: 2*(x+y)
alpha = 0.1
theta_x = 30
theta_y = 30
path_x = np.array([])
path_y = np.array([])
for i in range(20):
    print('x=%.2f, y=%.2f, theta_x=%.2f, theta_y=%.2f' 
          % (theta_x, theta_y, 
             fx(theta_x, theta_y), fy(theta_x, theta_y)))
    theta_x = theta_x - alpha * fx(theta_x, theta_y)
    theta_y = theta_y - alpha * fy(theta_x, theta_y)
    path_x = np.append(path_x, theta_x)
    path_y = np.append(path_y, theta_y)
    
x = np.repeat(np.arange(0, 30, 0.1), 300)
y = np.tile(np.arange(0, 30, 0.1), 300)
z = f(x, y)    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x, y, z, c=z, cmap='Greens', s=0.1, alpha=0.1, zorder=-1)
ax.scatter3D(path_x, path_y, f(path_x, path_y), 
             c='red', s=100, alpha=1,  zorder=1)

#

#%%
 
# Rosenbrock function
a = 1
b = 100
f = lambda x, y: (a - x)**2 + b * (y - x**2)**2
# the true minimum is at (1, 1)

fx = lambda x, y: 2*(x-a) + 4*b*x*(x**2-y)
fy = lambda x, y: 2*b*(y-x**2) 

def ff(theta):
    return np.array([fx(theta[0], theta[1]), 
                     fy(theta[0], theta[1])])



#%% 
# Teste Ableitungen
xx = np.arange(-2, 2, 0.01)
plt.plot(xx, (1-xx)**2+100*xx**4, xx, 2*(xx-1)+400*xx**3)
yy = np.array([0]*400)
plt.plot(xx, f(xx, yy), xx, fx(xx, yy))



#%%
# 
alpha = 0.0001 # learning rate
theta = np.array([3, 0])
path_x = np.array([])
path_y = np.array([])

for iter in range(3000):
    print('theta_x=%.2f, theta_y=%.2f' 
          % (theta[0], theta[1]))
    theta = theta - alpha * ff(theta)
    path_x = np.append(path_x, theta[0])
    path_y = np.append(path_y, theta[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.repeat(np.arange(-2, 2, 0.01), 400)
y = np.tile(np.arange(-1, 3, 0.01), 400)
z = f(x, y)    
ax.set_xlim([-2, 2.0])                                                       
ax.set_ylim([-1, 3.0])                                                       
ax.set_zlim([0, 2500]) 
ax.set_xlabel('x')
ax.set_xlabel('y')

ax.scatter3D(x, y, z, c=z, cmap='Greens', s=0.1, alpha=0.2, zorder=-1)
ax.scatter3D(path_x, path_y, f(path_x, path_y), 
             c='red', s=50, alpha=1,  zorder=1)


