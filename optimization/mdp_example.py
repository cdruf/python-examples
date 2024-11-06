#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDP aus Puterman, Beispiel 3.1
"""
import gurobipy as grb
import numpy as np

# States
S = [1, 2]

# Actions
A = grb.tuplelist([
    (1, 1), (1, 2),
    (2, 1)])

# Rewards
r = {A[0]: 5, A[1]: 10, A[2]: -1}

# Transition probabilities p(j | s, a), Indizes in Reihenfolge s, a, j
p = {(A[0], 1): 0.5, (A[0], 2): 0.5, (A[1], 1): 0, (A[1], 2): 1, (A[2], 1): 0, (A[2], 2): 1}

λ = 0.5
MDP = [S, A, r, p, λ]

# Print MDP
print("States: ", S)
for s in S:
    print("Actions in state", s)
    print(A.select(s, '*'))
    for a in A.select(s, '*'):
        print("Reward for ", a, "=", r[a])
        print("Transition probabilities")
        for j in S:
            print("p(", j, "|", a, ") =", p[(a, j)])
print("λ = ", λ)

# Politik
d = {}
A.select(2, '*')[0]


def policy_evaluation(S, r, p, λ, d):
    Pd = np.zeros((len(S), len(S)), dtype=float)
    rd = np.zeros((len(S), 1), dtype=float)
    for sInd in range(len(S)):
        s = sInd + 1
        a = d[s]
        for jInd in range(len(S)):
            j = jInd + 1
            Pd[sInd, jInd] = p[(a, j)]
        rd[sInd] = r[d[s]]
        # print (Pd)
    # print (cd)
    assert np.logical_and(0.99 < np.sum(Pd, 1), np.sum(Pd, 1) < 1.01).all()
    v = np.linalg.solve(np.eye(len(S)) - λ * Pd, rd)
    return v


def policy_improvement(S, A, r, p, λ, v):
    dn = {}
    v = np.zeros((len(S), 1))
    for sInd in range(len(S)):
        s = sInd + 1
        maxs = -10000000000
        for a in A.select(s, '*'):
            sumj = r[a]
            for j in S:
                jInd = j - 1
                sumj += λ * p[(a, j)] * v[jInd]
            if sumj > maxs:
                maxs = sumj
                argmaxs = a
        dn[s] = argmaxs
        v[sInd] = maxs
    return dn, v


def policy_iteration(S, A, r, p, λ):
    # initialisiere Politik mit beliebiger Entscheidungsregel
    d = {}
    for s in S:
        d[s] = A.select(s, '*')[0]

    n = 1
    while True:
        print("iteration", n)

        # Policy evaluation
        v = policy_evaluation(S, r, p, λ, d)

        # Policy improvement
        dn = policy_improvement(S, A, r, p, λ, v)[0]

        # Stop?
        if d == dn: return d, v
        d = dn


d, v = policy_iteration(S, A, r, p, λ)
print(d)
print(v)
