# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:08:15 2020

@author: 49856
"""

from time import time

import numpy as np
import pandas as pd


#%%
## Data
df = pd.read_csv("./demographic data.csv")
df

n = len(df)

m = 3
gu = np.array([10, 20, 20, n-9])
gl = np.array([3, 3, 3, 0])

K = 13
is_avg = np.array([0,1,1,0,1,1,1,1,1,0,0,0,0]).astype(bool)
P = df.values[:, 1:].astype(float)

stds = P.std(axis=0)

weights = np.ones(K)

# should the normalization be different for the sums?

#%%
# Simple data

#n = 4 # 
#m = 2 #
#K = 2 # number of features
#
#gu = np.array([100, 100, 100])
#gl = np.array([1, 1, 0])
#
#
#is_avg = np.array([0,1]).astype(bool)
#P = np.array([[1,0.1],
#              [2,0.2],
#              [3,0.3],
#              [4,0.4]])
#
#stds = P.std(axis=0)
#
#weights = np.ones(K)


#%%

def size_violations(x):
    """ Including un-assigned group. """
    sizes = np.bincount(x, minlength=m+1) 
    return (np.clip(sizes - gu, 0, np.Inf).sum() + 
            np.clip(gl - sizes, 0, np.Inf).sum())

def set_of_assigned_jobs(x, group):
    return np.where(x == group)[0]
    
def sets_of_assigned_jobs(x):
    return [set_of_assigned_jobs(x, i) for i in range(m+1)]

def metrics_assigned_jobs(x):
    ret = P[x != m, :].sum(axis=0) 
    ret[is_avg] /= (x != m).sum() 
    ret[~is_avg] /= m 
    return ret
        
def metrics_groups(x):
    """ Return matrix with the metrics (sum/avg) for each group (rows) and feature (cols). """
    s = sets_of_assigned_jobs(x)
    # if a group is empty there is division by 0
    ret = np.zeros((m,K))
    for g in range(m):
        ret[g, :] = P[s[g]].sum(axis=0)
        ret[g, is_avg] /= len(s[g]) 
        assert len(ret[g,:]) == K
    return ret

def calc_obj(x):
    # "metric" because either sum or avg
    metrics_a = metrics_assigned_jobs(x) # a
    metrics_g = metrics_groups(x) # ga
    assert (metrics_g.max(axis=0) - metrics_g.min(axis=0) >= 0).all()
    range_by_mean = ((metrics_g.max(axis=0) - metrics_g.min(axis=0)) / metrics_a)
    range_by_mean /= stds
    return range_by_mean.dot(weights)

#%%


class Individual:
    """
    Representation of a solution that can be used in genetic algorithms and
    other heuristics, and that consists of an array of the following form.

    +-------+-----------------+-----+-----------------+
    | Index | Item 1          | ... | Item n          |
    +-------+-----------------+-----+-----------------+
    | Value | Group of item 1 | ... | Group of item n |
    +-------+-----------------+-----+-----------------+
    
    Group m is the group with not assigned items.

    """

    best = None
    improvement = False
    
    @staticmethod
    def best_known_value():
        return (np.NaN if Individual.best is None
                else Individual.best.fitness)

    def __init__(self, randomize=False, assignment=None):
        if randomize:
            self.gene = np.random.choice(np.arange(m+1), n, p=gu/gu.sum())
            self.set_fitness_values()
        elif assignment is not None:
            assert assignment.dtype == 'int64'
            self.gene = assignment
            self.set_fitness_values()
        else:
            self.gene = np.empty(n, dtype=int)
            self.fitness = np.Inf
            # set used and fitness values when gene is initialized

    def set_fitness_values(self):
        vios = size_violations(self.gene)
        if vios == 0:
            self.fitness = calc_obj(self.gene)
        else: 
            self.fitness = 100 * vios
        self.update_best()
        
    
    def shift(self, j, σj):
        """ Re-assign j to σj. """
        self.gene[j] = σj
        self.set_fitness_values()
    
    def swap(self, j1, j2):
        """ Assign j1 to the current agent of j2 and vice versa. """
        # sizes stay the same
        # change assignments 
        self.gene[j1], self.gene[j2] = self.gene[j2], self.gene[j1]
        # change fitness (OPTIONAL: make more efficient)
        self.set_fitness_values()
    
    def random_shift(self):
        """ Randomly change a single assignment. """
        self.shift(np.random.randint(0,n), np.random.randint(0,m+1))
 
    def random_swap(self):
        self.swap(np.random.randint(0,n), np.random.randint(0,n))

   
    def update_best(self):
        if Individual.best is None or self.fitness < Individual.best.fitness:
            Individual.best = self
            Individual.best.time_created = time()
            Individual.improvement = True
        else:
            pass

    def __str__(self):
        return (str(self.gene) + ', '
                + str(self.fitness))

    def __eq__(self, other):
        return np.array_equal(self.gene, other.gene)

    def __hash__(self) -> int:
        return hash(np.array2string(self.gene))

    def __gt__(self, ind2):
        return self.fitness > ind2.fitness

    def __lt__(self, ind2):
        return self.fitness < ind2.fitness


#%%



def binary_tournament_selection(pop):
    """ Select two parents via binary tournaments. """
    N = len(pop)
    ind1 = np.random.randint(0, N)
    while True:
        ind2 = np.random.randint(0, N)
        if ind2 != ind1: break
    father = pop[ind1] if pop[ind1].fitness < pop[ind2].fitness else pop[ind2]

    while True:
        ind3 = np.random.randint(0, N)
        if ind3 not in (ind1, ind2): break
    while True:
        ind4 = np.random.randint(0, N)
        if ind4 not in (ind1, ind2, ind3): break
    mother = pop[ind3] if pop[ind3].fitness < pop[ind4].fitness else pop[ind4]

    return mother, father


def crossover_2_point(mama: Individual, papa: Individual):
    """ Create two new solutions by crossing two parent solutions """
    p1, p2 = get_2_crossover_points()
    child1 = Individual()
    child2 = Individual()
    for j in range(p1):
        child1.gene[j] = mama.gene[j]
        child2.gene[j] = papa.gene[j]
    for j in range(p1, p2):
        child1.gene[j] = papa.gene[j]
        child2.gene[j] = mama.gene[j]
    for j in range(p2, n):
        child1.gene[j] = mama.gene[j]
        child2.gene[j] = papa.gene[j]
    child1.set_fitness_values()
    child2.set_fitness_values()
    return child1, child2

def get_2_crossover_points():
    p1 = np.random.randint(0, n)
    while True:
        p2 = np.random.randint(0, n)
        if p1 != p2: break
    return min(p1, p2), max(p1, p2) # order them




#%%

def spaws_2(pop, prob_mutation=0.1):
    mother, father = binary_tournament_selection(pop)
    child1, child2 = crossover_2_point(mother, father)

    if np.random.rand() < prob_mutation: child1.random_shift()
    if np.random.rand() < prob_mutation: child2.random_shift()

    #child1.local_improvement()
    #child2.local_improvement()
    
    return child1, child2


def solve_ga(pop_size=1000, prob_mutation=0.1, 
             generations=500000, L=500, timeout_sec=600):
    """
    Genetic algorithm based on Chu and Beasley (1997) with some modifications.
    N - pop size
    M - stop when that many non-duplicate children have been generated
    L - stop when no improvement for that many new individuals
    """

    Individual.best = None
    no_improvement = 0
    start = time()
    
    # collectors for examining the convergence
    times = []
    best_value = []

    # Generate initial population
    pop = [Individual(True) for i in range(pop_size)]

    iteration = 0
    while iteration < generations:

        if time() - start > timeout_sec:
            print('timeout')
            break
        
        if no_improvement > L:
            print('no improvemt for %d iterations' % no_improvement)
            break
       
        if iteration % 100 == 0:
            print("iteration %d, no improvement for %d iterations" % 
                  (iteration, no_improvement))
#            for i in range(4):
#                print(pop[i])

        new_pop = [indi for i in range(pop_size//2) for indi in spaws_2(pop)]
        pop = new_pop
        
        if Individual.improvement:
            no_improvement = 0
        else:
            no_improvement += 1
        Individual.improvement = False
        
        # collect stats
        times.append(time() - start)
        best_value.append(Individual.best_known_value())
        
        iteration += 1

    for indi in pop:
        indi.update_best()
    
    print(Individual.best)

    return (np.NaN if Individual.best is None else Individual.best.fitness,
            time() - start,
            times, best_value)

#%%
## Execution
solve_ga()
