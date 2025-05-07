import pulp as pl

# Create an example problem with 2 columns and 2 rows
mdl = pl.LpProblem("MyLP", pl.LpMaximize)
xs = pl.LpVariable.dicts('x', indices=range(2), cat=pl.LpInteger)
mdl += pl.lpSum((i + 1) * x for i, x in xs.items())
mdl += pl.lpSum(x for x in xs.values()) <= 5, "constraint_1"
mdl += xs[1] <= 4, "constraint_2"
print(mdl)
mdl.writeMPS("my_example_mip.mps")
mdl.writeLP("my_example_mip.lp")

solver = pl.GUROBI(timeLimit=120, mipgap=0.0001)
mdl.solve(solver)
print(f"Objective value = {pl.value(mdl.objective):.0f}")
print(f"Gap = {mdl.solverModel.mipgap}")
print(f"Obj bound = {mdl.solverModel.objbound}")
print(f"x_0 = {pl.value(xs[0])}")
print(f"x_1 = {pl.value(xs[1])}")
