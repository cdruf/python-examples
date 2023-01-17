import pulp as pl

solvers_list = pl.listSolvers(onlyAvailable=True)
print(solvers_list)

# Build dummy model
mdl = pl.LpProblem('NiniModel', pl.LpMaximize)
xs = pl.LpVariable.dicts(name='x', indices=range(3), cat=pl.LpContinuous)
mdl += pl.lpSum((x for x in xs.values()))
mdl += pl.lpSum((x for x in xs.values())) <= 2, "c"
print(mdl)

# Solve with GUROBI_CMD
assert 'GUROBI_CMD' in solvers_list
solver_gb_cmd = pl.getSolver('GUROBI_CMD')
mdl.solve(solver_gb_cmd)
print(mdl.status)

# Solve with GUROBI
assert 'GUROBI' in solvers_list
# solver_gb = pl.getSolver('GUROBI')
solver_gb = pl.GUROBI()
# solver_gb.buildSolverModel(mdl)  # deprecated
mdl.solve(solver_gb)
print(mdl.solverModel.status)  # 2 for optimal

# Determine IIS with GUROBI
mdl = pl.LpProblem('NiniModel', pl.LpMaximize)
x = pl.LpVariable(name='x', cat=pl.LpContinuous, lowBound=0)
mdl += x <= -1, "c"
print(mdl)
mdl.solve(solver_gb)
print(mdl.solverModel.status)  # 3 for infeasible
mdl.solverModel.computeIIS()
mdl.solverModel.write("mini-model.ilp")
