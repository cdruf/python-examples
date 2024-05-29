import platform
from pathlib import Path

import pulp as pl

print(platform.uname().machine)
print(pl.listSolvers(onlyAvailable=True))
print('')

prob = pl.LpProblem("Test_LP", pl.LpMaximize)
x = pl.LpVariable('x', lowBound=0, cat='Continuous')
prob += x, "Z"
prob += x <= 2
prob += x >= 3
print(prob)

path_to_cplex = r'/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex'
print(Path(path_to_cplex).exists())

#
solver = pl.CPLEX_PY()
# solver.buildSolverModel(prob)
status = prob.solve(solver)
print(f'Status: {status}')
prob.solverModel.conflict.refine()
prob.solverModel.conflict.write("iis.lp")

# # Solve with CPLEX_PY - allows for computing IIS for example
# solver_cplex_py = pl.CPLEX_PY()
# prob.solve(solver_cplex_py)
# # prob.solverModel
# print(pl.LpStatus[prob.status])
# print(pl.value(prob.objective))
