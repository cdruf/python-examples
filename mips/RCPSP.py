"""
Resource constrained project scheduling problem.

Applications:
+ production management
+ make-to-order production
+ flow shop
+ job shop scheduling problem

"""
# %%
from collections import defaultdict

import numpy as np
import pulp
from pulp import lpSum


# %%

def get_planning_horiton(d_j):
    return d_j.sum()


def get_predecessors_and_followes(n, A):
    P = {i: [] for i in range(n)}
    F = {i: [] for i in range(n)}
    for i, j in A:
        P[j].append(i)
        F[i].append(j)
    return P, F


def get_earliest_and_latest_finish(T, n, d_j, A):
    EF_i = np.zeros(n, dtype=int)
    LF_i = np.repeat(T, n)
    return EF_i, LF_i


class Data:

    @staticmethod
    def get_default_instance():
        n = 5
        d_j = np.array([0, 2, 6, 7, 0])
        T = get_planning_horiton(d_j)
        A = {(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)}
        λ = defaultdict(lambda: 0)
        m = 2
        a_r = np.array([3, 4])
        u_jr = np.array([[0, 0],
                         [1, 1],
                         [2, 1],
                         [2, 3],
                         [0, 0]])
        return Data(T, n, d_j, A, λ, m, a_r, u_jr)

    def __init__(self, T, n, d_j, A, λ, m, a_r, u_jr):
        """
        Parameter
        ---------
        T : int
            Planning horizon
        n : int
            Number of jobs
        d_j : np.array
            Job durations
        A : Set[Tuple[Int, Int]]
            Precedence relationships
        λ : Set[Tuple[Int, Int]]
            Minimum timelag between 2 jobs in number of periods
        m : int
            Number of renewable resource types
        a_r : np.array
            Constant per period availability of resources
        u_jr : np.array
            Job's resource usage per period

        """
        self.T = T
        self.n = n
        self.d_j = d_j
        self.A = A
        self.λ = λ
        self.m = m
        self.a_r = a_r
        self.u_jr = u_jr

        self.P, self.F = get_predecessors_and_followes(n, A)
        """
        P : Map[int, List[int]]
            Set of predecessors
        F : Map[int, List[int]]
            Set of successers
        """
        self.EF_i, self.LF_i = get_earliest_and_latest_finish(T, n, d_j, A)
        """
        EF_j : np.array
            Earliest finish times of jobs
        LF_j : np.array
            Latest finish times of jobs
        """

    def __repr__(self):
        return f"n={self.n},d_n={self.d_j},A={self.A}"


# %%

data = Data.get_default_instance()

rcpsp = pulp.LpProblem("RCPSP", pulp.LpMinimize)

# Variable x_jt == 1, iff job j is finished at time t; 0 otherwise
x = {i: pulp.LpVariable.dicts(f"x_{i}", range(data.EF_i[i], data.LF_i[i]), cat=pulp.LpBinary)
     for i in range(data.n)}

# Objective function
rcpsp += lpSum([x for x in x[data.n - 1].values()]), "z"

# Constraints
# Finish all projects
for i in range(data.n):
    rcpsp += lpSum([x for x in x[i].values()]) == 1, f"c_finish_{i}"

# Precendence relations
for j, predecessors in data.P.items():
    lhs = lpSum([(t - data.d_j[j]) * x[j][t] for t in x[j].keys()])
    for i in predecessors:
        rhs = lpSum([t * x[i][t] for t in x[i].keys()])
        rcpsp += lhs >= rhs, f"c_pred_{j},{i}"

# Resource restrictions


print(rcpsp)

rcpsp.solve()

print(pulp.LpStatus[rcpsp.status])
print(pulp.value(rcpsp.objective))

for variable in rcpsp.variables():
    print("{} = {}".format(variable.name, variable.varValue))
