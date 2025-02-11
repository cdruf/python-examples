"""
HCOM: assignment 2.1, patient flow
"""

import gurobipy as grb


class Arc:

    def __init__(self, pre, succ, lag):
        self.pre = pre
        self.succ = succ
        self.lag = lag

    def __str__(self):
        return '(' + str(self.pre) + ',' + str(self.lag) + ',' + str(self.succ) + ')'


class TimeWindow:
    def __init__(self, E, L):
        self.E = E
        self.L = L

    def __str__(self):
        return '{' + str(self.E) + ', ..., ' + str(self.L) + '}'


# Parameter

P = 2  # No. patients; p = 0, ..., P-1
T = 6  # No. periods; t = 0, ..., T-1
A = 6  # No. activities; i = 0, ..., A-1

Rd = 3  # No. day resources; k = 0, ..., Rd-1
Rn = 2  # No. night resources (wards); k = 0, ..., Rn-1

# capacity of day resources (Indices k, t)
RD = [[1, 1, 2, 2, 2, 2],
      [1, 1, 2, 2, 2, 2],
      [1, 1, 2, 2, 2, 2]]

# number of beds in wards (Indices k, t) 
RN = [[1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 0, 0]]

# capacity demand of activities (Indices i, k)
rD = [[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]]

b = [0, 1]  # ward of each patient

# contribution margin for p, if he has a LOS of 1, ..., 6 days 
π = [[1700, 1950, 1900, 1850, 2810, 1825],
     [2150, 2400, 2100, 2050, 2050, 2030]]

# 0 -> 1 -> 2 and 3 -> 4 -> 5
clinicalPathways = [Arc(0, 1, 0),
                    Arc(1, 2, 3),
                    Arc(3, 4, 0),
                    Arc(4, 5, 3)]
print(*clinicalPathways, sep="\n")

α = [0, 1]  # admission period of each p
φ = [2, 5]  # discharge activity of each p

# Time windows for each activity
W = [TimeWindow(0, 5),
     TimeWindow(0, 5),
     TimeWindow(0, 5),
     TimeWindow(1, 5),
     TimeWindow(1, 5),
     TimeWindow(1, 5)]
print(*W, sep="\n")


# helper function for keys of x and variable names
def key(i, j):
    return str(i) + "," + str(j)



# Create model
model = grb.Model("model")

# Create variables
x = {}
for i in range(A):
    for t in range(W[i].E, W[i].L + 1):
        x[key(i, t)] = model.addVar(vtype=grb.GRB.BINARY, name="x_" + key(i, t))

# Set objective
expr = 0
for p in range(P):
    i = φ[p]  # index of discharge activity
    for t in range(W[i].E, W[i].L + 1):  # time window
        LOT = t - α[p] + 1
        reward = π[p][LOT - 1]
        assert reward > 0
        expr += reward * x[key(i, t)]
model.setObjective(expr, grb.GRB.MAXIMIZE)

# Add constraints
for e in clinicalPathways:
    expr = 0
    j = e.succ
    for t in range(W[j].E, W[j].L + 1):
        expr += t * x[key(j, t)]
    i = e.pre
    for t in range(W[i].E, W[i].L + 1):
        expr += -t * x[key(i, t)]
    model.addConstr(expr >= e.lag, "cTimeLag_" + key(i, j))

for k in range(Rd):
    for t in range(T):
        expr = 0
        for i in range(A):
            if W[i].E <= t and t <= W[i].L:
                expr += rD[i][k] * x[key(i, t)]
        model.addConstr(expr <= RD[k][t], "cDayResources_" + key(k, t))

for k in range(Rn):
    for t in range(T):
        expr = 0
        RHS = RN[k][t]
        for p in range(P):
            if b[p] == k and t >= α[p]:
                RHS -= 1
                i = φ[p]  # discharge activity index
                for τ in range(W[i].E, min(t, W[i].L + 1)):
                    expr -= x[key(i, τ)]
        model.addConstr(expr <= RHS, "cNightResources_" + key(k, t))

for i in range(A):
    expr = 0
    for t in range(W[i].E, W[i].L + 1):
        expr += x[key(i, t)]
    model.addConstr(expr == 1, "cEachActivity_" + str(i))

model.write("model.lp")  # write model to file

model.optimize()  # optimize

# Print results
print('\nSolution')
print('Obj: %g' % model.objVal)
for i in range(A):
    for t in range(W[i].E, W[i].L + 1):
        if x[key(i, t)].s == 1:
            print("Activity " + str(i) + " is performed in period " + str(t))
