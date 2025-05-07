# -*- coding: utf-8 -*-
"""
Facility location problem.
@author: Christian Ruf
"""

from dataclasses import dataclass
from time import time

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import GRB

# %%
# Data

EPS = 0.0001  # floating point precision


def euclidean(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# %%
# Instance


class Instance:

    def __init__(self, m, custs_x, custs_y, demands,
                 n, locs_x, locs_y, loc_costs, capacities,
                 distances):
        assert type(m) == int
        self.m = m
        assert len(custs_x) == len(custs_y) == len(demands) == m
        self.custs = range(m)
        self.custs_x = custs_x
        self.custs_y = custs_y
        self.demands = demands
        assert type(n) == int
        self.n = n
        assert len(locs_x) == len(locs_y) == n
        assert len(loc_costs) == len(capacities) == n
        self.locs = range(n)
        self.locs_x = locs_x
        self.locs_y = locs_y
        assert loc_costs.dtype == float
        self.loc_costs = loc_costs
        self.capacities = capacities
        assert distances.shape == (n, m)
        self.distances = distances

    @classmethod
    def random(cls, m=15, n=5):
        custs_x = np.random.randint(0, 100, m)
        custs_y = np.random.randint(0, 100, m)
        locs_x = np.random.randint(0, 100, n)
        locs_y = np.random.randint(0, 100, n)
        distances = np.array([euclidean(custs_x[i], custs_y[i],
                                        locs_x[j], locs_y[j])
                              for j in range(n) for i in range(m)]).reshape(n, m)
        # add some noise representing wiggly streets
        distances += (1 + np.random.random((n, m))) * distances
        return cls(m, custs_x, custs_y,
                   np.random.randint(0, 100, m),
                   n, locs_x, locs_y,
                   np.random.randint(0, 200, n).astype(float),
                   np.array([10000 for j in range(n)]),
                   distances)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.n != other.n or self.m != other.m:
            return False
        if not (self.custs_x == other.custs_x).all():
            return False
        if not (self.custs_y == other.custs_y).all():
            return False
        if not (self.demands == other.demands).all():
            return False
        if not (self.locs_x == other.locs_x).all():
            return False
        if not (self.locs_y == other.locs_y).all():
            return False
        if not (self.loc_costs == other.loc_costs).all():
            return False
        if not (self.capacities == other.capacities).all():
            return False
        if not (self.distances == other.distances).all():
            return False

    def __str__(self):
        return str(self.__dict__)


# %%

def generate_instances():
    return [Instance.random(25 + 2 * i, 5 + i) for i in range(10)]


# %%
@dataclass
class Solution:
    status: int
    value: float
    secs: float
    y: np.array
    xs: np.array


def mycallback(model, where):
    if where == GRB.Callback.POLLING:
        pass
    elif where == GRB.Callback.PRESOLVE:
        cdels = model.cbGet(GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print('%d columns and %d rows are removed' % (cdels, rdels))
    elif where == GRB.Callback.MIPSOL:
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        bnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        print('New solution at node %d, obj %g, bnd %g, sol %d ' %
              (nodecnt, obj, bnd, solcnt))
        model._times.append(time() - model._start)
        model._best_value.append(obj)
        g = abs(bnd - obj) / (bnd + EPS)


class Model(object):
    def __init__(self, data: Instance, timeout_sec=120):
        self.data = data

        # Build model
        self.m = gp.Model()

        # initialize fields required in callback
        self.m._times = []
        self.m._best_value = []
        self.m._start = time()

        self.timeout = timeout_sec
        self.y = self.m.addVars(data.n, vtype=GRB.BINARY, name="y")
        self.x = self.m.addVars(data.n, data.m, vtype=GRB.CONTINUOUS, name="x")
        self.m.setObjective(
            gp.quicksum(data.loc_costs[j] * self.y[j] for j in data.locs) +
            gp.quicksum(data.distances[j, i] * self.x[j, i] for j in data.locs for i in data.custs),
            GRB.MINIMIZE)
        self.m.addConstrs((self.x._sum('*', i) >= data.demands[i]
                           for i in data.custs), "demand")
        self.m.addConstrs((self.x._sum(j, '*') <= data.capacities[j] * self.y[j]
                           for j in data.locs), "capacity")
        self.m.update()
        # self.m.write("./facility.lp")

    def solve(self):
        self.m.setParam('TimeLimit', self.timeout - (time() - self.m._start))
        self.m.optimize(mycallback)
        self.m.optimize()

        print("\nStatus: %d" % self.m.status)
        if self.m.status == GRB.OPTIMAL:
            print('Obj: %g' % self.m.objVal)
            ys = self.m.getAttr('x', self.y)
            output_data = [(self.y[j].varName, ys[j]) for j in self.data.locs if ys[j] > 0.00]
            xs = self.m.getAttr('x', self.x)
            output_data += [(self.x[j, i].varName, xs[j, i]) for j in self.data.locs for i in self.data.custs if
                            xs[j, i] > 0]
            output_df = pd.DataFrame(data=output_data, columns=['Variable', 'Value'])
            print(output_df)
            return Solution(self.m.status, self.m.objVal, time() - self.m._start, ys, xs)
        else:
            raise RuntimeError("Model not solved")


# %%
# Visualization with matplotlib

def draw_figure(dat: Instance, sol: Solution):
    locs_volume = [sol.xs._sum(j, '*').getValue() for j in dat.locs]
    colors = [j for j in dat.locs]
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.scatter(dat.locs_x, dat.locs_y, c=['black'] * data.n, s=[5] * data.n)
    ax.scatter(dat.locs_x, dat.locs_y, c=colors, s=locs_volume)
    for i in dat.custs:
        for j in dat.locs:
            if sol.xs[j, i] > 0:
                plt.plot([dat.locs_x[j], dat.custs_x[i]],
                         [dat.locs_y[j], dat.custs_y[i]])
                plt.annotate(str(sol.xs[j, i]),
                             xy=((dat.locs_x[j] + dat.custs_x[i]) / 2,
                                 (dat.locs_y[j] + dat.custs_y[i]) / 2))
    ax.grid(True)


# %%
"""
# Experiments 
"""
if __name__ == '__main__':
    data = Instance.random()
    model = Model(data)
    solution = model.solve()
    draw_figure(data, solution)
    plt.show()
    plt.tight_layout()
