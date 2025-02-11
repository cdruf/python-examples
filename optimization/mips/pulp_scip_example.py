import pulp as pl

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
print("\n\n")

# Solve with SCIP
solver = pl.SCIP_CMD(path="/Users/christian/Downloads/SCIPOptSuite-8.0.3-Darwin/bin/scip")
prob.solve(solver)
print(pl.LpStatus[prob.status])
print(pl.value(prob.objective))
