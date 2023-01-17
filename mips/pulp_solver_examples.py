
import pulp as pl
from pathlib import Path
from pprint import pprint

prob = pl.LpProblem("Test_LP", pl.LpMaximize)
x = pl.LpVariable('x', lowBound=0, cat='Continuous')
prob += x, "Z"
prob += x <= 2
print(prob)

print(pl.listSolvers(onlyAvailable=False))
print(pl.listSolvers(onlyAvailable=True))

# Solve with standard CBC solver
solver_cbc = pl.getSolver('PULP_CBC_CMD', msg=False)
prob.solve(solver=solver_cbc)
print(pl.LpStatus[prob.status])
print(pl.value(prob.objective))

# Solve with CPLEX_CMD
# path_to_cplex = r'/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/cplex'
path_to_cplex = r'/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex'
solver_cplex = pl.CPLEX_CMD(path=path_to_cplex, keepFiles=True)
prob.solve(solver_cplex)
print(pl.LpStatus[prob.status])
print(pl.value(prob.objective))

# Solve with CPLEX_PY - allows for computing IIS for example
print("### CPLEX_PY ### ")

import sys
cplex_py_path = r'/Applications/CPLEX_Studio221/cplex/python/3.9/x86-64_osx'
print(Path(cplex_py_path).exists())
sys.path.append(cplex_py_path)
sys.path.append(r'/Applications/CPLEX_Studio221/cplex/python')
sys.path.append(r'/Applications/CPLEX_Studio221/cplex')
sys.path.append(r'/Applications/CPLEX_Studio221/python')



pprint(sys.path)
import cplex
solver_cplex_py = pl.CPLEX_PY()
prob.solve(solver_cplex_py)
#prob.solverModel
print(pl.LpStatus[prob.status])
print(pl.value(prob.objective))



