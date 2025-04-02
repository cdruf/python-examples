import pulp as pl

# Create model
prob = pl.LpProblem("Test_LP", pl.LpMaximize)
x = pl.LpVariable('x', lowBound=0, cat='Continuous')
prob += x, "Objective"
prob += x <= 2, "Constraint"

# Solve
solver = pl.HiGHS(timeLimit=100, gapRel=0.001, msg=True)
status = prob.solve(solver)
print(f"\nHiGHS Status: {pl.LpStatus[status]}")
print(f"Objective value: {pl.value(prob.objective)}")
