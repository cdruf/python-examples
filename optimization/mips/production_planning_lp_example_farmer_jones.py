"""
This is an implementation of the Farmer Jones problem from
Winston - Operations Research: Applications and Algorithms.
"""
import pulp

mdl = pulp.LpProblem("farmer_jones_model", pulp.LpMaximize)

x1 = pulp.LpVariable('x1', lowBound=0, cat='Continuous')
x2 = pulp.LpVariable('x2', lowBound=0, cat='Continuous')

# Objective function
mdl += 3 * x1 + 4 * x2, "z"

# Constraints
mdl += x1 >= 30, "min_qty"
mdl += 1 / 10 * x1 + 1 / 25 * x2 <= 7, "surface"
mdl += 4 / 10 * x1 + 10 / 25 * x2 <= 40, "work"

print(mdl)

mdl.solve()

print('\n=== Solution ===\n')

print(pulp.LpStatus[mdl.status])
print("Obj. value =", pulp.value(mdl.objective))
for variable in mdl.variables():
    print(f"{variable.name} = {variable.varValue}")
for name, constraint in mdl.constraints.items():
    print(f"slack({name}) = {constraint.slack}")
