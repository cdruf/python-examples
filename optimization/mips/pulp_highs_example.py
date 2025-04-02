import highspy
import pulp as pl

# Create model
prob = pl.LpProblem("Test_LP", pl.LpMaximize)
x = pl.LpVariable('x', lowBound=0, cat='Continuous')
prob += x, "Z"
prob += x <= 2, "ub"

# Solve with HiGHS
prob.writeLP("problem.lp")  # write LP
h = highspy.Highs()
h.readModel("problem.lp")
h.run()

# Get the solution
status = h.getModelStatus()
value = h.getObjectiveValue()
solution = h.getSolution()
var_values = solution.col_value
x_val = var_values[0]

print(f"Status: {status}")
print(f"Objective: {value}")
print(f"Variable value: {x_val}")
