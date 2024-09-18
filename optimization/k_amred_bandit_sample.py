#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit sample.

@author: Christian
"""



from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

#%%
"""
# Test normal distribution
"""
distr = norm(5, 1) 
samples = [distr.rvs() for _ in range(1000)]
samples
plt.hist(samples)


#%%

class Data:
    
    @staticmethod
    def get_10_armed_default_instance():
        k = 10
        distributions = [norm(norm.rvs(1), 1) for _ in range(k)]
        return Data(k, distributions)
    
    def __init__(self, k, distributions):
        self.k = k
        self.distributions = distributions
        self.means = np.array([distributions[i].mean() for i in range(k)])
        self.optimal_action = self.means.argmax()
        
    def __str__(self):
        return f"k={self.k}," + ",".join(
            [f"mean_{i}={self.means[i]:.4f}" for i in range(self.k)])

    def __repr__(self):
        return self.__str__()
    
    def sample(self, a):
        return self.distributions[a].rvs()
    
        
        

#%%

instances = [Data.get_10_armed_default_instance() for _ in range(100)] 


#%% 

def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item 
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        if q_values[i] > top_value: 
            top_value = q_values[i]
            ties = [i]
        elif q_values[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)


# %%

"""
# Basic algorithm
"""

def solve(instance, iterations=1000, epsilon=0.1):
    Q = np.repeat(10.0, instance.k)  # optimistic estimates
    N = np.zeros(instance.k, dtype=int)  # number of evaluations
    
    total_reward = 0.0
    share_best_action = [0]
    rms_error = [(((instance.means - Q) ** 2).sum() / instance.k) ** 0.5]
    
    for iteration in range(iterations):
        if np.random.rand() >= epsilon:
            A = argmax(Q)
        else:
            A = np.random.randint(instance.k)
        R = instance.sample(A)
        N[A] += 1
        Q[A] = Q[A] + 1/N[A] * (R - Q[A])
        
        total_reward += R
        tmp = share_best_action[iteration] 
        tmp2 = A == instance.optimal_action
        share_best_action.append(tmp + 1.0/(iteration+1) * (tmp2-tmp))
        rms_error.append(
            (((instance.means - Q) ** 2).sum() / instance.k) ** 0.5)

    return Q, share_best_action, rms_error

#%%

Q1, s1, rms1 = solve(instances[0], iterations=5000)
Q2, s2, rms2 = solve(instances[0], iterations=5000, epsilon=0.01)

#%% 

plt.plot(range(5001), s1, 'r-', 
         range(5001), s2, 'b--')

#%%
plt.plot(range(5001), rms1, 'r-',
         range(5001), rms2, 'b--')

