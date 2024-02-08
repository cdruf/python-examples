#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized assignment problem and various algorithms.
"""
import random as rd
import time
import math
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb

from sortedcontainers import SortedSet



#%%
"""
Parameter (small test instance)
"""


n = 10 # No. items / jobs
m = 4 # No. buckets / agents

# capacity of buckets
b = np.array([rd.randint(1, 10) for j in range(m)])

# profits / costs
c = np.array([[rd.randint(1, 10) for j in range(n)] for i in range(m)])

# weights
a = np.array([[rd.randint(0, 10) for j in range(n)] for i in range(m)])


#%%

""" Instances """

class Instance:

    def __init__(self, n, m, a, b, c):
        self.n = n
        self.m = m
        self.a = a
        assert a.shape == (m, n)
        self.b = b
        assert len(self.b.shape) == 1
        assert self.b.shape[0] == m
        self.c = c
        assert c.shape == (m, n)

    def to_global_vars(self):
        global n
        n = self.n
        global m
        m = self.m
        global a
        a = self.a
        global b
        b = self.b
        global c
        c = self.c

# Instances based on Cattrysse (1994)
def generate_cat(n, m):
    a = np.array([[rd.randint(5, 25) for j in range(n)] for i in range(m)])
    c = np.array([[rd.randint(15, 25) for j in range(n)] for i in range(m)])
    b = np.array([0.8 / m * a[i, :].sum() for i in range(m)])
    return Instance(n, m, a, b, c)
M_VALS = [5, 8, 10]
N_by_M = [3, 4, 5, 6]    
insts_cat = [generate_cat(m*r, m) for r in N_by_M for m in M_VALS]


# Test instances based on Özbakir (2009).
def gen_a_c(n, m):
    a = np.array([[rd.randint(5, 25) for j in range(n)] for i in range(m)])
    c = np.array([[rd.randint(10, 50) for j in range(n)] for i in range(m)])
    return a, c

def Ij(c, j):
    """ Return an agent index with minimum costs for job j. """
    return np.where(c[:, j] == np.amin(c[:, j]))[0][0]

def r_sum(n, a, c, i):
    return reduce(lambda x, y: x+y, [a[i][j] if Ij(c, j) == i else 0 for j in range(n)]) 

def R(n, m, a, c):
    return max([r_sum(n, a, c, i) for i in range(m)])

def generate_type_A(n, m):
    a, c = gen_a_c(n, m)
    bb = 0.6 * n / m * 15 + 0.4 * R(n, m, a, c) 
    b = np.array([bb for i in range(m)])
    return Instance(n, m, a, b, c)

def generate_type_B(n, m):
    a, c = gen_a_c(n, m)
    bb = (0.6 * n / m * 15 + 0.4 * R(n, m, a, c)) * 0.7
    b = np.array([bb for i in range(m)])
    return Instance(n, m, a, b, c)

def generate_type_C(n, m):
    # same as Cattrysse
    a, c = gen_a_c(n, m)
    b = np.array([0.8 / m * a[i, :].sum() for i in range(m)])
    return Instance(n, m, a, b, c)

def generate_type_D(n, m):
    a = np.array([[rd.randint(1, 100) for j in range(n)] for i in range(m)])
    c = np.array([[111 - a[i][j] + rd.randint(-10, 10) for j in range(n)] for i in range(m)]) 
    b = np.array([0.8 / m * a[i, :].sum() for i in range(m)])
    return Instance(n, m, a, b, c)

N_M_COMB = [(n,m) for n in [100, 200] for m in [5, 10, 20]]

insts_oez_type_A = [generate_type_A(n, m) for n, m in N_M_COMB]
insts_oez_type_B = [generate_type_B(n, m) for n, m in N_M_COMB]
insts_oez_type_C = [generate_type_C(n, m) for n, m in N_M_COMB]
insts_oez_type_D = [generate_type_D(n, m) for n, m in N_M_COMB]



#%%


def mycallback(model, where):
    if where == grb.GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == grb.GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(grb.GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(grb.GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print('%d columns and %d rows are removed' % (cdels, rdels))
    elif where == grb.GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(grb.GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(grb.GRB.Callback.MIPSOL_SOLCNT)
        print('New solution at node %d, obj %g, sol %d ' % (nodecnt, obj, solcnt))
        model._times.append(time.time() - model._start)
        model._best_value.append(obj)




def solve_model(timeout_sec=60):
    ''' Solve the problem with the Gurobi solver '''

    start = time.time()

    # build model
    model = grb.Model("model")
    x = model.addVars(m, n, vtype=grb.GRB.INTEGER, ub=1, name="x")
    indices = grb.tuplelist([(i, j) for i in range(m) for j in range(n)])
    model.setObjective(grb.quicksum(c[i][j]*x[i, j] for i, j in indices),
                       grb.GRB.MINIMIZE)
    model.addConstrs((grb.quicksum(a[i][j]*x[i, j] for j in range(n)) <= b[i]
                      for i in range(m)), name='capacity')
    model.addConstrs((grb.quicksum(x[i, j] for i in range(m)) == 1
                      for j in range(n)), name='assign')
    model.update()

    # collectors for examining the convergence
    model._times = []
    model._best_value = []

    # Optimize
    model.write("assignment.lp")
    model.setParam('TimeLimit', timeout_sec - (time.time() - start))
    model._start = start
    model.optimize(mycallback)

    print('\nStatus: %d' % model.status)
    print('Obj: %g' % model.objVal)
    for j in range(n):
        for i in range(m):
            if x[i, j].s > 0.001:
                print("x_%d,%d = %f" % (i, j, x[i, j].s))

    # return optimal value and computational time in secs
    return (model.status, 
            model.objVal, 
            time.time() - start, 
            model._times, 
            model._best_value)

#%%


class Individual:
    """
    Representation of a solution that can be used in genetic algorithms and
    other heuristics, and that consists of an array of the following form.

    Index: | job 1           | ... | job n           |
    Value: | bucket of job 1 | ... | bucket of job n |

    The assignment constraints are automatically satisfied, as each job is
    assigned (unless NaN values are allowed).
    The capacity constraints may be violated.
    """

    best = None
    childs_with_no_improvement = 0
    
    def best_known_value():
        return (np.NaN if Individual.best is None or Individual.best.unfitness > 0.001 
                else Individual.best.fitness)

    def __init__(self, randomize=False, assignment=None):
        if randomize:
            self.gene = np.array([rd.randint(0, m-1) for j in range(n)])
            self.set_fitness_values()
        elif assignment is not None:
            assert assignment.dtype == 'int64'
            self.gene = assignment
            self.set_fitness_values()
        else:
            self.gene = np.zeros(n, dtype=int)
            self.fitness = np.Inf
            self.unfitness = np.Inf

    def get_used_capacities(self):
        used = np.zeros(m)
        for j in range(n):
            used[self.gene[j]] += a[self.gene[j]][j]
        return used

    def get_used_capacities_and_sets_of_assigned_jobs(self):
        used = np.zeros(m)
        sets = [[] for i in range(m)]
        for j in range(n):
            used[self.gene[j]] += a[self.gene[j]][j]
            sets[self.gene[j]].append(j)
        return used, sets


    def set_fitness_values(self):
        self.set_fitness()
        self.set_unfitness()

    def set_fitness(self):
        self.fitness = 0.0
        for j in range(n):
            self.fitness += c[self.gene[j]][j]

    def set_unfitness(self):
        self.unfitness = np.clip(self.get_used_capacities() - b, 0, np.Inf).sum()

    def mutate(self):
        j1 = rd.randint(0, n-1)
        j2 = rd.randint(0, n-1)
        self.gene[j1], self.gene[j2] = self.gene[j2], self.gene[j1]

    def local_improvement(self):
        self.local_improvement_a()
        self.local_improvement_b()

    def local_improvement_a(self):
        """
        If the resource capacity of a bucket i is exceeded,
        select a random job j of the jobs assigned to bucket i and
        re-assign job j to another bucket, if its capacity is not exceeded then.
        """
        used, sets = self.get_used_capacities_and_sets_of_assigned_jobs()
        for i in range(m):
            if used[i] > b[i]:
                j = sets[i][rd.randint(0, len(sets[i])-1)]
                self.local_improvement_a_reassign(used, i, j)


    def local_improvement_a_reassign(self, used, Tj, j):
        k = (Tj+1) % m
        while k != Tj:
            if b[k] >= used[k] + a[k][j]:
                self.gene[j] = k
                return
            k = (k+1) % m

    def local_improvement_b(self):
        """
        Make a feasible reassignment that improves the total cost.
        """
        used = self.get_used_capacities()
        for j in range(n):
            Tj = self.gene[j]
            min_c = c[Tj][j]
            argmin = None
            k = (Tj+1) % m
            while k != Tj:
                if c[k][j] < min_c and used[k] + a[k][j] <= b[k]:
                    min_c = c[k][j]
                    argmin = k
                k = (k+1) % m
            if argmin is not None:
                self.gene[j] = argmin
                used = self.get_used_capacities()

    def random_local_move_1(self):
        """
        Change a single assignment.assignment
        """
        self.gene[rd.randint(0, n-1)] = rd.randint(0, m-1)

    def random_local_move_2(self):
        """
        Exchange 2 assignments.
        """
        self.mutate()

    def update_best(self):
        assert self.fitness < np.Inf and self.unfitness < np.Inf
        if Individual.best is None or self < Individual.best:
            Individual.best = self
            Individual.best.time_created = time.time()
            Individual.childs_with_no_improvement = 0
        else:
            Individual.childs_with_no_improvement += 1


    def __str__(self):
        return (str(self.gene) + ', '
                + str(self.fitness) + ', '
                + str(self.unfitness))

    def __eq__(self, other):
        return np.array_equal(self.gene, other.gene)

    def __hash__(self) -> int:
        return hash(np.array2string(self.gene))

    def __gt__(self, ind2):
        if self.unfitness > ind2.unfitness:
            return True
        if self.unfitness < ind2.timeoutunfitness:
            return False
        return self.fitness > ind2.fitness

    def __lt__(self, ind2):
        if self.unfitness < ind2.unfitness:
            return Trueassignment
        if self.unfitness > ind2.unfitness:
            return False
        return self.fitness < ind2.fitness




def binary_tournament_selection(pop):
    """
    Select two parents via binary tournaments,
    where only fitness is considered, i.e. ignoring unfitness.
    """
    N = len(pop)
    ind1 = rd.randint(0, N-1)
    while True:
        ind2 = rd.randint(0, N-1)
        if ind2 != ind1: break
    father = pop[ind1] if pop[ind1].fitness < pop[ind2].fitness else pop[ind2]

    while True:
        ind3 = rd.randint(0, N-1)
        if ind3 not in (ind1, ind2): break
    while True:
        ind4 = rd.randint(0, N-1)
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
    return child1, child2

def get_2_crossover_points():
    p1 = rd.randint(0, n-1)
    while True:
        p2 = rd.randint(0, n-1)
        if p1 != p2: break
    return min(p1, p2), max(p1, p2) # order them




#%%

def solve_ga(N=100, M=500000, L=10000, prob_mutation=0.5, timeout_sec=60):
    """
    Genetic algorithm based on Chu and Beasley (1997) with some modifications.
    N - pop size
    M - stop when that many non-duplicate children have been generated
    L - stop when no improvement for that many new individuals
    """

    Individual.best = None
    start = time.time()
    
    # collectors for examining the converassignmentgence
    times = []
    best_value = []

    # Generate initial population
    # pop = SortedSet([Individual(True) for i in range(N)])
    pop = SortedSet()
    while len(pop) < N:
        individual = Individual(True)
        if individual not in pop:
            individual.update_best()
            pop.add(individual)

    #print(pop[1])
    #print(pop[N-1])

    Individual.childs_with_no_improvement = 0
    iteration = 0
    while iteration < M:

        if time.time() - start > timeout_sec:
            print('timeout')
            break
        
        if iteration % 500 == 0:
            print(iteration)

        mother, father = binary_tournament_selection(pop)
        child1, child2 = crossover_2_point(mother, father)

        if rd.random() < prob_mutation: child1.mutate()
        if rd.random() < prob_mutation: child2.mutate()

        child1.local_improvement()
        child2.local_improvement()

        child1.set_fitness_values()
        child2.set_fitness_values()

        if child1 not in pop:
            del pop[N-1]
            pop.add(child1)
            iteration += 1
            # print(child1)
            child1.update_best()

        if child2 not in pop:
            del pop[N-1]
            pop.add(child2)
            iteration += 1
            # print(child2)
            child2.update_best()

        if Individual.childs_with_no_improvement > L:
            print('no improvemt for %d iterations' % Individual.childs_with_no_improvement)
            break
        
        # collect stats
        times.append(time.time() - start)
        best_value.append(Individual.best_known_value())

    print(pop[0])
    assert Individual.best == pop[0]
    print(pop[1])

    return (np.NaN if pop[0].unfitness > 0 else pop[0].fitness,
            time.time() - start,
            times, best_value,)

#%%

def greedy():
    """ Construct an initial solution greedily """
    ret = Individual()
    for j in range(n):
        used = ret.get_used_capacities()
        mini = np.Inf
        argmin = None
        for i in range(m):
            violation = max(used[i] + a[i][j] - b[i], 0)
            cost = c[i][j] + 10**6 * violation
            if cost < mini:
                mini = cost
                argmin = i
        ret.gene[j] = argmin
    ret.set_fitness_values()
    return ret

def grasp():
    """ 
    Radomized greedy. 
    """
    gene = np.zeros(n, dtype=int)
    sets = [[] for i in range(m)] # assigned jobs for each agent
    used = np.zeros(m) # used capacity for each agent
    for j in np.random.permutation(n):
        costs = np.array([c[i][j] + 10**2 * max(used[i] + a[i][j] - b[i], 0) for i in range(m)])
        zaehler = 1.0
        nenner = 0.0
        for i in range(m):
            zaehler *= costs[i]
            for l in range(i+1, m):
                nenner += costs[i]*costs[l]
        p = np.array([zaehler / costs[i] / nenner for i in range(m)])
        assert p.sum() == 1
        p = np.cumsum(p)
        assert 0.9999 < p[-1] < 1.0001
        Tj = np.argmax(rd.random() <= p)
        gene[j] = Tj
        used[Tj] += a[Tj][j]
        sets[Tj].append(j)
    ret = Individual(assignment=gene)
    ret.set_fitness_values()
    return ret
            
        
        
grasp()

    
#%%

def solve_sa(cooling_rate = 0.9999, temperature = 4 * n, timeout_sec=60):
    """
    Simulated annealing.
    """
    Individual.best = None
    start = time.time()
    
    def worsening(x, x_new):
        """ infeasibility should be weighted more """
        if x_new.unfitness > x.unfitness:
            return (x_new.unfitness - x.unfitness) * 10
        return x_new.fitness - x.fitness

    def acceptance_probability(worsening):
        assert worsening >= 0
        return math.exp(-worsening/temperature)

    # collectors for examining the convergence
    times = []
    best_value = []
    current_value = []

    # random initial solution
    x = greedy()

    while temperature > 0.01:
        print('temperature = %.2f, value of best known solution = %.0f' % 
              (temperature, Individual.best_known_value()))
        
        if time.time() - start > timeout_sec:
            print('timeout')
            break
        
        # random neighborhood move
        x_new = Individual(assignment=np.copy(x.gene))
        move = rd.randint(0, 1)
        if move == 1:
            x_new.random_local_move_1()
        else:
            x_new.random_local_move_2()
        x_new.set_fitness_values()

        # move or not
        if x_new < x:
            x = x_new
            x.update_best()
        else:
            if rd.random() < acceptance_probability(worsening(x, x_new)):
                x = x_new

        # reduce temperature
        temperature *= cooling_rate
        
        # collect stats
        times.append(time.time() - start)
        best_value.append(Individual.best_known_value())
        current_value.append(np.NaN if x.unfitness > 0 else x.fitness)

    return (Individual.best_known_value(),
            time.time() - start,
            times, best_value, current_value)


#%%


def solve_ba(N=10, M=5, e=2, nsp=1, nep=3):
    """ 
    Bee algorithm.
    N - # scout bees,
    M - # selected patches,
    e - # best patches,
    nep - # bees recruited for the E best patches,
    nsp - # bees recruited for the other M - e patches,
    ngh - size of patches.
    """    

#%%

insts_oez_type_A[-1].to_global_vars()
insts_oez_type_B[-1].to_global_vars()
insts_oez_type_C[-1].to_global_vars()
insts_oez_type_D[-1].to_global_vars()

m_status, m_value, m_secs, m_times, m_best_value = solve_model()
ga_value, ga_secs, ga_times, ga_best_value = solve_ga()
sa_value, sa_secs, sa_times, sa_best_value, sa_current_value = solve_sa()

plt.plot(sa_times, sa_best_value, 'r-')
# plt.plot(sa_times, sa_current_value, 'm.')
plt.plot(ga_times, ga_best_value, 'b-')
plt.axhline(m_value, color='black')
plt.axvline(m_secs, color='black')
plt.plot(m_times, m_best_value, 'bo')


#%%

g_feinunze = 31.1034768
1356 * 100 / g_feinunze

for j in np.random.permutation(n):
    print(j)
